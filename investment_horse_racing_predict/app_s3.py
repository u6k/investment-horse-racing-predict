import os
import io
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
    obj = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=os.getenv("S3_FOLDER")+s3_key)
    df = pd.read_csv(obj["Body"], **kwargs)

    return df


def write_dataframe(df, s3_key):
    with io.StringIO() as buf:
        df.to_csv(buf)
        s3 = get_s3_client()
        s3.put_object(
            Bucket=os.getenv("S3_BUCKET"),
            Key=os.getenv("S3_FOLDER")+s3_key,
            Body=io.BytesIO(buf.getvalue().encode())
        )
