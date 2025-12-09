import inspect
import logging
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import yaml
from rich.console import Console
from typer import Option, Typer

from ursa.cli.config import Settings

app = Typer()


def _parameter_defaults(func) -> dict[str, Any]:
    """Return a mapping of parameter defaults for a callable."""

    signature = inspect.signature(func)
    defaults: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.default is not inspect._empty:
            defaults[name] = param.default
    return defaults


def _build_settings(
    config_file: Optional[Path],
    cli_values: dict[str, Any],
    defaults: dict[str, Any],
) -> Settings:
    """Load settings from YAML (if provided) and overlay CLI overrides."""

    file_values: dict[str, Any] = {}
    if config_file is not None:
        config_path = config_file.expanduser()
        if not config_path.exists():
            logging.exception(f"Config file '{config_path}' not found.")

        try:
            loaded = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            logging.exception(
                f"Failed to read config file '{config_path}': {exc}"
            )
        if loaded is None:
            file_values = {}
        elif isinstance(loaded, dict):
            file_values = loaded
        else:
            logging.exception(
                "The YAML configuration must contain a top-level mapping."
            )

    settings = Settings(**file_values)
    overrides = {
        key: value
        for key, value in cli_values.items()
        if key not in defaults or value != defaults[key]
    }
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings


@app.command(help="Start ursa REPL")
def run(
    workspace: Annotated[
        Path, Option(help="Directory to store ursa ouput")
    ] = Path("ursa_workspace"),
    config_file: Annotated[
        Optional[Path],
        Option(
            "--config",
            "-c",
            help="Path to YAML settings file.",
            envvar="URSA_CONFIG_FILE",
        ),
    ] = None,
    llm_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and name of LLM to use for agent tasks. "
                "Use format <provider>:<model-name>. "
                "For example 'openai:gpt-5'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
            envvar="URSA_LLM_NAME",
        ),
    ] = "openai:gpt-5",
    llm_base_url: Annotated[
        Optional[str],
        Option(help="Base url for LLM.", envvar="URSA_LLM_BASE_URL"),
    ] = None,
    llm_api_key: Annotated[
        Optional[str],
        Option(help="API key for LLM", envvar="URSA_LLM_API_KEY"),
    ] = None,
    max_completion_tokens: Annotated[
        int, Option(help="Maximum tokens for LLM to output")
    ] = 50000,
    emb_model_name: Annotated[
        Optional[str],
        Option(
            help=(
                "Model provider and Embedding model name. "
                "Use format <provider>:<embedding-model-name>. "
                "For example, 'openai:text-embedding-3-small'. "
                "See: https://reference.langchain.com/python/langchain/embeddings/?h=init_embeddings#langchain.embeddings.init_embeddings"
            ),
            envvar="URSA_EMB_NAME",
        ),
    ] = None,
    emb_base_url: Annotated[
        Optional[str],
        Option(help="Base url for embedding model", envvar="URSA_EMB_BASE_URL"),
    ] = None,
    emb_api_key: Annotated[
        Optional[str],
        Option(help="API key for embedding model", envvar="URSA_EMB_API_KEY"),
    ] = None,
    share_key: Annotated[
        bool,
        Option(
            help=(
                "Whether or not the LLM and embedding model share the same "
                "API key. If yes, then you can specify only one of them."
            )
        ),
    ] = False,
    thread_id: Annotated[
        str,
        Option(help="Thread ID for persistance", envvar="URSA_THREAD_ID"),
    ] = "ursa_cli",
    safe_codes: Annotated[
        list[str],
        Option(
            help="Programming languages that the execution agent can trust by default.",
            envvar="URSA_THREAD_ID",
        ),
    ] = ["python", "julia"],
    arxiv_summarize: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to summarize response."
        ),
    ] = True,
    arxiv_process_images: Annotated[
        bool,
        Option(help="Whether or not to allow ArxivAgent to process images."),
    ] = False,
    arxiv_max_results: Annotated[
        int,
        Option(
            help="Maximum number of results for ArxivAgent to retrieve from ArXiv."
        ),
    ] = 10,
    arxiv_database_path: Annotated[
        Optional[Path],
        Option(
            help="Path to download/downloaded ArXiv documents; used by ArxivAgent."
        ),
    ] = None,
    arxiv_summaries_path: Annotated[
        Optional[Path],
        Option(help="Path to store ArXiv paper summaries; used by ArxivAgent."),
    ] = None,
    arxiv_vectorstore_path: Annotated[
        Optional[Path],
        Option(
            help="Path to store ArXiv paper vector store; used by ArxivAgent."
        ),
    ] = None,
    arxiv_download_papers: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to download ArXiv papers."
        ),
    ] = True,
    ssl_verify_llm: Annotated[
        bool, Option(help="Whether or not to verify SSL certificates for LLM.")
    ] = True,
    ssl_verify_emb: Annotated[
        bool,
        Option(
            help="Whether or not to verify SSL certificates for embedding model."
        ),
    ] = True,
) -> None:
    console = Console()
    with console.status("[grey50]Loading ursa ..."):
        from ursa.cli.hitl import HITL, UrsaRepl

    cli_values = {
        "workspace": workspace,
        "llm_model_name": llm_model_name,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "max_completion_tokens": max_completion_tokens,
        "emb_model_name": emb_model_name,
        "emb_base_url": emb_base_url,
        "emb_api_key": emb_api_key,
        "share_key": share_key,
        "safe_codes": safe_codes,
        "thread_id": thread_id,
        "arxiv_summarize": arxiv_summarize,
        "arxiv_process_images": arxiv_process_images,
        "arxiv_max_results": arxiv_max_results,
        "arxiv_database_path": arxiv_database_path,
        "arxiv_summaries_path": arxiv_summaries_path,
        "arxiv_vectorstore_path": arxiv_vectorstore_path,
        "arxiv_download_papers": arxiv_download_papers,
        "ssl_verify_llm": ssl_verify_llm,
        "ssl_verify_emb": ssl_verify_emb,
    }

    settings = _build_settings(
        config_file=config_file,
        cli_values=cli_values,
        defaults=_parameter_defaults(run),
    )
    hitl = HITL.from_settings(settings)
    UrsaRepl(hitl).run()


@app.command()
def version() -> None:
    from ursa import __version__

    print(__version__)


@app.command(help="Start MCP server to serve ursa agents")
def serve(
    config_file: Annotated[
        Optional[Path],
        Option(
            "--config",
            "-c",
            help="Path to YAML settings file.",
            envvar="URSA_CONFIG_FILE",
        ),
    ] = None,
    transport: Annotated[
        Literal["stdio", "sse", "streamable-http"],
        Option(
            "--transport",
            "-t",
            case_sensitive=False,
            help="Transport to expose the MCP server on",
        ),
    ] = "stdio",
    host: Annotated[
        str,
        Option(
            "--host",
            help="Host to bind for network transports (ignored for stdio)",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        Option(
            "--port",
            help="Port to bind for network transports (ignored for stdio)",
        ),
    ] = 8000,
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO",
) -> None:
    console = Console()
    with console.status("[grey50]Starting ursa MCP server ..."):
        from ursa.cli.hitl import HITL

    settings = _build_settings(config_file=config_file)
    hitl = HITL.from_settings(settings)
    mcp = hitl.as_mcp_server(host=host, port=port, log_level=log_level)
    mcp.run(transport=transport)


def main():
    app()
