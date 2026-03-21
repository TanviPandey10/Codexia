from dotenv import load_dotenv
import os

# FIXED: Looking for .env in the same directory as the script
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import argparse
import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
import re
import sys
from datetime import datetime, timedelta

from github import Github, Auth  # Added Auth for modern GitHub API support
from gitlab import Gitlab
from langchain_community.callbacks.manager import get_openai_callback

from codedog.actors.reporters.pull_request import PullRequestReporter
from codedog.chains import CodeReviewChain, PRSummaryChain
from codedog.retrievers import GithubRetriever, GitlabRetriever
from codedog.utils.langchain_utils import load_model_by_name
from codedog.utils.email_utils import send_report_email
from codedog.utils.git_hooks import install_git_hooks
from codedog.utils.git_log_analyzer import get_file_diffs_by_timeframe, get_commit_diff, CommitInfo
from codedog.utils.code_evaluator import DiffEvaluator, generate_evaluation_markdown


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CodeDog - AI-powered code review tool")

    # Main operation subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # PR review command
    pr_parser = subparsers.add_parser("pr", help="Review a GitHub or GitLab pull request")
    pr_parser.add_argument("repository", help="Repository path (e.g. owner/repo)")
    pr_parser.add_argument("pr_number", type=int, help="Pull request number to review")
    pr_parser.add_argument("--platform", choices=["github", "gitlab"], default="github",
                         help="Platform to use (github or gitlab, defaults to github)")
    pr_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")
    pr_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")

    # Setup git hooks command
    hook_parser = subparsers.add_parser("setup-hooks", help="Set up git hooks for commit-triggered reviews")
    hook_parser.add_argument("--repo", help="Path to git repository (defaults to current directory)")

    # Developer code evaluation command
    eval_parser = subparsers.add_parser("eval", help="Evaluate code commits of a developer in a time period")
    eval_parser.add_argument("author", help="Developer name or email (partial match)")
    eval_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD), defaults to 7 days ago")
    eval_parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to today")
    eval_parser.add_argument("--repo", help="Git repository path or name (e.g. owner/repo for remote repositories)")
    eval_parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    eval_parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    eval_parser.add_argument("--model", help="Evaluation model, defaults to CODE_REVIEW_MODEL env var or gpt-3.5")
    eval_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")
    eval_parser.add_argument("--output", help="Report output path, defaults to codedog_eval_<author>_<date>.md")
    eval_parser.add_argument("--platform", choices=["github", "gitlab", "local"], default="local",
                         help="Platform to use (github, gitlab, or local, defaults to local)")
    eval_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")

    # Commit review command
    commit_parser = subparsers.add_parser("commit", help="Review a specific commit")
    commit_parser.add_argument("commit_hash", help="Commit hash to review")
    commit_parser.add_argument("--repo", help="Git repository path or name (e.g. owner/repo for remote repositories)")
    commit_parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    commit_parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    commit_parser.add_argument("--model", help="Review model, defaults to CODE_REVIEW_MODEL env var or gpt-3.5")
    commit_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")
    commit_parser.add_argument("--output", help="Report output path, defaults to codedog_commit_<hash>_<date>.md")
    commit_parser.add_argument("--platform", choices=["github", "gitlab", "local"], default="local",
                         help="Platform to use (github, gitlab, or local, defaults to local)")
    commit_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")

    return parser.parse_args()


def parse_emails(emails_str: Optional[str]) -> List[str]:
    """Parse comma-separated email addresses."""
    if not emails_str:
        return []

    return [email.strip() for email in emails_str.split(",") if email.strip()]


def parse_extensions(extensions_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated file extensions."""
    if not extensions_str:
        return None

    return [ext.strip() for ext in extensions_str.split(",") if ext.strip()]


async def pr_summary(retriever, summary_chain):
    """Generate PR summary asynchronously."""
    result = await summary_chain.ainvoke(
        {"pull_request": retriever.pull_request}, include_run_info=True
    )
    return result


async def code_review(retriever, review_chain):
    """Generate code review asynchronously."""
    result = await review_chain.ainvoke(
        {"pull_request": retriever.pull_request}, include_run_info=True
    )
    return result


def get_remote_commit_diff(
    platform: str,
    repository_name: str,
    commit_hash: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    gitlab_url: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Get commit diff from remote repositories (GitHub or GitLab).
    """
    if platform.lower() == "github":
        # Initialize GitHub client with modern Auth
        github_token = os.getenv("GITHUB_TOKEN")
        auth = Auth.Token(github_token) if github_token else None
        github_client = Github(auth=auth)
        
        print(f"Analyzing GitHub repository {repository_name} for commit {commit_hash}")

        try:
            repo = github_client.get_repo(repository_name)
            commit = repo.get_commit(commit_hash)

            file_diffs = {}
            for file in commit.files:
                _, ext = os.path.splitext(file.filename)

                if include_extensions and ext not in include_extensions:
                    continue
                if exclude_extensions and ext in exclude_extensions:
                    continue

                if file.patch:
                    file_diffs[file.filename] = {
                        "diff": f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}",
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                    }

            return file_diffs

        except Exception as e:
            error_msg = f"Failed to retrieve GitHub commit: {str(e)}"
            print(error_msg)
            return {}

    elif platform.lower() == "gitlab":
        # Initialize GitLab client
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            error_msg = "GITLAB_TOKEN environment variable is not set"
            print(error_msg)
            return {}

        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")

        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} for commit {commit_hash}")

        try:
            project = gitlab_client.projects.get(repository_name)
            commit = project.commits.get(commit_hash)
            diff = commit.diff()

            file_diffs = {}
            for file_diff in diff:
                file_path = file_diff.get("new_path", "")
                old_path = file_diff.get("old_path", "")
                diff_content = file_diff.get("diff", "")

                if not diff_content:
                    continue

                _, ext = os.path.splitext(file_path)

                if include_extensions and ext not in include_extensions:
                    continue
                if exclude_extensions and ext in exclude_extensions:
                    continue

                if file_diff.get("new_file", False):
                    status = "A"
                elif file_diff.get("deleted_file", False):
                    status = "D"
                else:
                    status = "M"

                formatted_diff = f"diff --git a/{old_path} b/{file_path}\n{diff_content}"

                additions = diff_content.count("\n+")
                deletions = diff_content.count("\n-")

                file_diffs[file_path] = {
                    "diff": formatted_diff,
                    "status": status,
                    "additions": additions,
                    "deletions": deletions,
                }

            return file_diffs

        except Exception as e:
            error_msg = f"Failed to retrieve GitLab commit: {str(e)}"
            print(error_msg)
            return {}

    else:
        error_msg = f"Unsupported platform: {platform}. Use 'github' or 'gitlab'."
        print(error_msg)
        return {}


def get_remote_commits(
    platform: str,
    repository_name: str,
    author: str,
    start_date: str,
    end_date: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    gitlab_url: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, Dict[str, str]], Dict[str, int]]:
    """
    Get commits from remote repositories (GitHub or GitLab).
    """
    if platform.lower() == "github":
        github_token = os.getenv("GITHUB_TOKEN")
        auth = Auth.Token(github_token) if github_token else None
        github_client = Github(auth=auth)
        
        print(f"Analyzing GitHub repository {repository_name} for commits by {author}")

        try:
            repo = github_client.get_repo(repository_name)

            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

            commits = []
            commit_file_diffs = {}

            all_commits = repo.get_commits(since=start_datetime, until=end_datetime)

            for commit in all_commits:
                if author.lower() in commit.commit.author.name.lower() or (
                    commit.commit.author.email and author.lower() in commit.commit.author.email.lower()
                ):
                    commit_info = CommitInfo(
                        hash=commit.sha,
                        author=commit.commit.author.name,
                        date=commit.commit.author.date,
                        message=commit.commit.message,
                        files=[file.filename for file in commit.files],
                        diff="\n".join([
                            f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}"
                            for file in commit.files if file.patch
                        ]),
                        added_lines=sum(file.additions for file in commit.files),
                        deleted_lines=sum(file.deletions for file in commit.files),
                        effective_lines=sum(file.additions - file.deletions for file in commit.files)
                    )
                    commits.append(commit_info)

                    file_diffs = {}
                    for file in commit.files:
                        if file.patch:
                            _, ext = os.path.splitext(file.filename)

                            if include_extensions and ext not in include_extensions:
                                continue
                            if exclude_extensions and ext in exclude_extensions:
                                continue

                            file_diffs[file.filename] = file.patch

                    commit_file_diffs[commit.sha] = file_diffs

            code_stats = {
                "total_added_lines": sum(commit.added_lines for commit in commits),
                "total_deleted_lines": sum(commit.deleted_lines for commit in commits),
                "total_effective_lines": sum(commit.effective_lines for commit in commits),
                "total_files": len(set(file for commit in commits for file in commit.files))
            }

            return commits, commit_file_diffs, code_stats

        except Exception as e:
            error_msg = f"Failed to retrieve GitHub commits: {str(e)}"
            print(error_msg)
            return [], {}, {}

    elif platform.lower() == "gitlab":
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            print("GITLAB_TOKEN not set")
            return [], {}, {}

        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")
        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} for commits by {author}")

        try:
            project = gitlab_client.projects.get(repository_name)
            commits = []
            commit_file_diffs = {}
            start_iso = f"{start_date}T00:00:00Z"
            end_iso = f"{end_date}T23:59:59Z"
            all_commits = project.commits.list(all=True, since=start_iso, until=end_iso)

            for commit in all_commits:
                if author.lower() in commit.author_name.lower() or (
                    commit.author_email and author.lower() in commit.author_email.lower()
                ):
                    commit_detail = project.commits.get(commit.id)
                    diff = commit_detail.diff()
                    file_diffs = {}
                    for file_diff in diff:
                        file_path = file_diff.get("new_path", "")
                        diff_content = file_diff.get("diff", "")
                        if not diff_content: continue
                        _, ext = os.path.splitext(file_path)
                        if include_extensions and ext not in include_extensions: continue
                        if exclude_extensions and ext in exclude_extensions: continue
                        file_diffs[file_path] = diff_content

                    if not file_diffs: continue
                    commit_info = CommitInfo(
                        hash=commit.id,
                        author=commit.author_name,
                        date=datetime.strptime(commit.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"),
                        message=commit.message,
                        files=list(file_diffs.keys()),
                        diff="\n\n".join(file_diffs.values()),
                        added_lines=sum(d.count("\n+") for d in file_diffs.values()),
                        deleted_lines=sum(d.count("\n-") for d in file_diffs.values()),
                        effective_lines=0
                    )
                    commits.append(commit_info)
                    commit_file_diffs[commit.id] = file_diffs

            code_stats = {"total_files": len(commits)}
            return commits, commit_file_diffs, code_stats
        except Exception as e:
            print(f"GitLab error: {e}")
            return [], {}, {}

    return [], {}, {}


async def evaluate_developer_code(
    author: str,
    start_date: str,
    end_date: str,
    repo_path: Optional[str] = None,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    model_name: str = "gpt-3.5",
    output_file: Optional[str] = None,
    email_addresses: Optional[List[str]] = None,
    platform: str = "local",
    gitlab_url: Optional[str] = None,
):
    """Evaluate a developer's code commits in a time period."""
    if not output_file:
        author_slug = author.replace("@", "_at_").replace(" ", "_").replace("/", "_")
        output_file = f"codedog_eval_{author_slug}.md"

    model = load_model_by_name(model_name)
    print(f"Evaluating {author}'s code commits...")

    if platform.lower() == "local":
        commits, commit_file_diffs, code_stats = get_file_diffs_by_timeframe(
            author, start_date, end_date, repo_path, include_extensions, exclude_extensions
        )
    else:
        commits, commit_file_diffs, code_stats = get_remote_commits(
            platform, repo_path, author, start_date, end_date, include_extensions, exclude_extensions, gitlab_url
        )

    if not commits:
        print("No commits found.")
        return

    evaluator = DiffEvaluator(model)
    with get_openai_callback() as cb:
        evaluation_results = await evaluator.evaluate_commits(commits, commit_file_diffs)
        report = generate_evaluation_markdown(evaluation_results)
        report += f"\n\nCost: ${cb.total_cost}"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_file}")
    return report


def generate_full_report(repository_name, pull_request_number, email_addresses=None, platform="github", gitlab_url=None):
    """Generate a full report including PR summary and code review."""
    start_time = time.time()

    if platform.lower() == "github":
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            print("ERROR: GITHUB_TOKEN not found in .env! Please check your file.")
            return "Error: Token missing"

        # FIXED AUTHENTICATION logic
        auth = Auth.Token(github_token)
        github_client = Github(auth=auth)

        print(f"Analyzing GitHub repository {repository_name} PR #{pull_request_number}")
        try:
            retriever = GithubRetriever(github_client, repository_name, pull_request_number)
            print(f"Successfully retrieved PR: {retriever.pull_request.title}")
        except Exception as e:
            print(f"Failed to retrieve GitHub PR: {str(e)}")
            return str(e)

    elif platform.lower() == "gitlab":
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")
        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        try:
            retriever = GitlabRetriever(gitlab_client, repository_name, pull_request_number)
        except Exception as e:
            return str(e)
    else:
        return "Unsupported platform"

    code_summary_model = os.environ.get("CODE_SUMMARY_MODEL", "gpt-3.5-turbo")
    pr_summary_model = os.environ.get("PR_SUMMARY_MODEL", "gpt-3.5-turbo")
    code_review_model = os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5-turbo")

    # --- ENHANCED CHAIN INITIALIZATION TO BYPASS VALIDATION ERROR ---
    summary_chain = None
    review_chain = None
    
    try:
        print("Loading AI Summary Chain...")
        summary_chain = PRSummaryChain.from_llm(
            code_summary_llm=load_model_by_name(code_summary_model),
            pr_summary_llm=load_model_by_name(pr_summary_model),
            verbose=True,
        )
    except Exception as e:
        print(f"\n[!] WARNING: Summary Chain could not be initialized due to Pydantic/LangChain version conflict: {e}")

    try:
        print("Loading AI Review Chain...")
        review_chain = CodeReviewChain.from_llm(
            llm=load_model_by_name(code_review_model),
            verbose=True,
        )
    except Exception as e:
        print(f"\n[!] WARNING: Review Chain could not be initialized: {e}")

    with get_openai_callback() as cb:
        pr_summary_val = "Summary analysis not available."
        code_summaries_val = {}
        code_reviews_val = []

        if summary_chain:
            try:
                res = asyncio.run(pr_summary(retriever, summary_chain))
                pr_summary_val = res.get("pr_summary", pr_summary_val)
                code_summaries_val = res.get("code_summaries", {})
            except Exception as e:
                print(f"Execution Error in Summary: {e}")

        if review_chain:
            try:
                res = asyncio.run(code_review(retriever, review_chain))
                code_reviews_val = res.get("code_reviews", [])
            except Exception as e:
                print(f"Execution Error in Review: {e}")

        reporter = PullRequestReporter(
            pr_summary=pr_summary_val,
            code_summaries=code_summaries_val,
            pull_request=retriever.pull_request,
            code_reviews=code_reviews_val,
            telemetry={"cost": cb.total_cost, "time": time.time()-start_time},
        )
        report = reporter.report()
        return report


def main():
    """Main function to parse arguments and run the appropriate command."""
    args = parse_args()

    if args.command == "pr":
        email_addresses = parse_emails(args.email or os.environ.get("NOTIFICATION_EMAILS", ""))
        report = generate_full_report(args.repository, args.pr_number, email_addresses, args.platform, args.gitlab_url)
        print("\n--- FINAL REVIEW REPORT ---\n")
        print(report)

    elif args.command == "setup-hooks":
        repo_path = args.repo or os.getcwd()
        install_git_hooks(repo_path)

    elif args.command == "eval":
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = args.start_date or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        asyncio.run(evaluate_developer_code(
            args.author, start_date, end_date, args.repo, parse_extensions(args.include),
            parse_extensions(args.exclude), args.model, args.output, parse_emails(args.email), args.platform, args.gitlab_url
        ))

if __name__ == "__main__":
    main()