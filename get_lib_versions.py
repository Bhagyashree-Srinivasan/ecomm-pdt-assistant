import importlib.metadata
packages = [
    "langchain",
    "langchain_core",
    "python-dotenv",
    "streamlit",
    "beautifulsoup4",
    "fastapi",
    "html5lib",
    "jinja2",
    "langchain-astradb",
    "langchain-google-genai",
    "langchain-groq",
    "lxml",
    "python-multipart",
    "selenium",
    "undetected-chromedriver",
    "uvicorn",
    "structlog"
]

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: (not installed)")