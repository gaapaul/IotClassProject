from google.cloud import storage

def upload_blob(bucket_name, blob_text, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(blob_text)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def log_data(request):
    request_json = request.get_json()
    BUCKET_NAME = 'my-bucket'
    BLOB_NAME = 'test-blob'
    BLOB_STR = '{"blob": "some json"}'

    upload_blob(BUCKET_NAME, BLOB_STR, BLOB_NAME)
    return f'Success!'