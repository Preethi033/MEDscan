def main() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "Uvicorn is required to run the API. Install with "
            "`python -m pip install '.[api]'`."
        ) from error
    uvicorn.run("scanexplain.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
