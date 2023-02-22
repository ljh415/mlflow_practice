import os
import base64
import argparse
import requests
import pandas as pd

# MODEL_SERVER_URI = "http://192.168.10.231:1234/invocations"

def pred(path, host, port):
    
    filenames = [path]
    if os.path.isdir(path):
        filenames = [
            os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))
        ]
    else :
        filenames = [path]
    
    def read_image(x):
        with open(x, "rb") as f:
            return f.read()

    data = pd.DataFrame(
        data=[base64.encodebytes(read_image(x)) for x in filenames], columns=["image"]
    ).to_json(orient="split")

    response = requests.post(
        url=f"http://{host}:{port}/invocations",
        data={
            "dataframe_split": data,
        },
        headers={"Content-Type": "application/json"},
    )
    
    if response.status_code != 200:
        raise Exception(
            f"Status Code {response.status_code}. {response.text}"
        )
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="1234")
    parser.add_argument("--path", type=str)
    
    args = parser.parse_args()