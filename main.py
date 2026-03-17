from fastapi import FastAPI


app = FastAPI()


@app.get("/hello/{input_text}")
def hello(input_text: str) -> dict[str, str]:
    return {"message": f"Hello, World {input_text}"}
