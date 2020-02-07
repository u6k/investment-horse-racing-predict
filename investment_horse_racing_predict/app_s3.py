import os
import io
import joblib
import pandas as pd
import boto3


def get_s3_client():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT")
    )

    return s3


def read_dataframe(s3_key, **kwargs):
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=os.getenv("S3_FOLDER")+"/"+s3_key)
    df = pd.read_csv(obj["Body"], **kwargs)

    return df


def write_dataframe(df, s3_key):
    with io.StringIO() as buf:
        df.to_csv(buf)
        s3 = get_s3_client()
        s3.put_object(
            Bucket=os.getenv("S3_BUCKET"),
            Key=os.getenv("S3_FOLDER")+"/"+s3_key,
            Body=io.BytesIO(buf.getvalue().encode())
        )


def read_sklearn_model(s3_key):
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=os.getenv("S3_FOLDER")+"/"+s3_key)
    with io.BytesIO(obj["Body"].read()) as buf:
        model = joblib.load(buf)

    return model


def write_sklearn_model(model, s3_key):
    with io.BytesIO() as buf:
        joblib.dump(model, buf, compress=9)
        s3 = get_s3_client()
        s3.put_object(
            Bucket=os.getenv("S3_BUCKET"),
            Key=os.getenv("S3_FOLDER")+"/"+s3_key,
            Body=io.BytesIO(buf.getvalue())
        )
