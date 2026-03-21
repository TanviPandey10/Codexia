from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
# Import BaseOutputParser but using Any for type checking to avoid Pydantic abstract error
from pydantic import ConfigDict, Field

from codedog.chains.pr_summary.prompts import CODE_SUMMARY_PROMPT, PR_SUMMARY_PROMPT
from codedog.models import ChangeSummary, PRSummary, PullRequest
from codedog.processors.pull_request_processor import (
    SUFFIX_LANGUAGE_MAPPING,
    PullRequestProcessor,
)

processor = PullRequestProcessor.build()


class PRSummaryChain(Chain):
    """Chain to summarize a pull request."""

    code_summary_chain: LLMChain = Field(exclude=True)
    pr_summary_chain: LLMChain = Field(exclude=True)
    # FIXED: Changed type hint to Any and default to None to bypass Abstract Class validation
    parser: Any = Field(default=None, exclude=True)

    _input_keys: List[str] = ["pull_request"]
    _output_keys: List[str] = ["pr_summary", "code_summaries"]

    # FIXED: Updated config for Pydantic V2 compatibility
    model_config = ConfigDict(
        extra="ignore",  # Changed from forbid to ignore to be safer
        arbitrary_types_allowed=True
    )

    @property
    def _chain_type(self) -> str:
        return "pull_request_summary_chain"

    @property
    def input_keys(self) -> List[str]:
        return self._input_keys

    @property
    def output_keys(self) -> List[str]:
        return self._output_keys

    def review(self, inputs: Dict[str, Any], _run_manager: CallbackManagerForChainRun) -> Dict[str, Any]:
        pr: PullRequest = inputs["pull_request"]

        code_summary_inputs = self._process_code_summary_inputs(pr)
        code_summary_outputs = (
            self.code_summary_chain.apply(
                code_summary_inputs,
                callbacks=_run_manager.get_child(tag="CodeSummary"),
            )
            if code_summary_inputs
            else []
        )

        code_summaries = processor.build_change_summaries(
            code_summary_inputs, code_summary_outputs
        )

        pr_summary_input = self._process_pr_summary_input(pr, code_summaries)
        pr_summary_output = self.pr_summary_chain(
            pr_summary_input,
            callbacks=_run_manager.get_child(tag="PRSummary"),
        )

        return self._process_result(pr_summary_output, code_summaries)

    async def areview(self, inputs: Dict[str, Any], _run_manager: AsyncCallbackManagerForChainRun) -> Dict[str, Any]:
        pr: PullRequest = inputs["pull_request"]

        code_summary_inputs = self._process_code_summary_inputs(pr)
        code_summary_outputs = (
            await self.code_summary_chain.aapply(
                code_summary_inputs, callbacks=_run_manager.get_child()
            )
            if code_summary_inputs
            else []
        )

        code_summaries = processor.build_change_summaries(
            code_summary_inputs, code_summary_outputs
        )

        pr_summary_input = self._process_pr_summary_input(pr, code_summaries)
        pr_summary_output = await self.pr_summary_chain.ainvoke(
            pr_summary_input, callbacks=_run_manager.get_child()
        )

        return await self._aprocess_result(pr_summary_output, code_summaries)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs["pull_request"].json() + "\n")
        return self.review(inputs, _run_manager)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs["pull_request"].json() + "\n")
        return await self.areview(inputs, _run_manager)

    def _process_code_summary_inputs(self, pr: PullRequest) -> List[Dict[str, str]]:
        input_data = []
        code_files = processor.get_diff_code_files(pr)

        for code_file in code_files:
            input_data.append(
                {
                    "content": code_file.diff_content.content[:2000],
                    "name": code_file.full_name,
                    "language": SUFFIX_LANGUAGE_MAPPING.get(code_file.suffix, ""),
                }
            )
        return input_data

    def _process_pr_summary_input(
        self, pr: PullRequest, code_summaries: List[ChangeSummary]
    ) -> Dict[str, str]:

        return {
            "change_files": processor.gen_material_change_files(pr.change_files),
            "code_summaries": processor.gen_material_code_summaries(code_summaries),
            "metadata": processor.gen_material_pr_metadata(pr),
        }

    def _process_result(
        self, pr_summary_output: Dict[str, Any], code_summaries: List[ChangeSummary]
    ) -> Dict[str, Any]:
        return {
            "pr_summary": pr_summary_output["text"],
            "code_summaries": code_summaries,
        }

    async def _aprocess_result(
        self, pr_summary_output: Dict[str, Any], code_summaries: List[ChangeSummary]
    ) -> Dict[str, Any]:

        raw_output_text = pr_summary_output.get("text", "[No text found]")
        logging.warning(f"Raw LLM output: {raw_output_text}")

        return {
            "pr_summary": raw_output_text,
            "code_summaries": code_summaries,
        }

    @classmethod
    def from_llm(
        cls,
        code_summary_llm: BaseLanguageModel,
        pr_summary_llm: BaseLanguageModel,
        code_summary_prompt: PromptTemplate = CODE_SUMMARY_PROMPT,
        pr_summary_prompt: PromptTemplate = PR_SUMMARY_PROMPT,
        **kwargs,
    ) -> "PRSummaryChain":

        parser = OutputFixingParser.from_llm(
            llm=pr_summary_llm,
            parser=PydanticOutputParser(pydantic_object=PRSummary),
        )

        code_summary_chain = LLMChain(
            llm=code_summary_llm,
            prompt=code_summary_prompt,
        )

        pr_summary_chain = LLMChain(
            llm=pr_summary_llm,
            prompt=pr_summary_prompt,
            output_parser=parser,
        )

        return cls(
            code_summary_chain=code_summary_chain,
            pr_summary_chain=pr_summary_chain,
            parser=parser,
            **kwargs,
        )