# A dummy mcp server for testing
from fastmcp.server import FastMCP

mcp = FastMCP()


@mcp.tool
def add(x: float, y: float) -> float:
    return x + y


if __name__ == "__main__":
    mcp.run(transport="stdio")
