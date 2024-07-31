import zipfile
import os
import requests


def fetch_zip_file(URL):
    # Try to get to the zip file
    try:
        response = requests.get(URL)
    except OSError:
        print("No connection to the server")
        return None

    # check the request
    if response.status_code == 200:
        # save the dataset
        open("sources.zip", "wb").write(response.content)
    else:
        print("Zip file request not successful!")
        return None


def download_and_unzip(URL):
    # fetch the zip file
    fetch_zip_file(URL)

    # unzip
    unzip_file = input("Where do you want to save the unzip file? ")
    with zipfile.ZipFile("sources.zip", "r") as zip_ref:
        zip_ref.extractall(unzip_file)
        print("Download successful!")

    return unzip_file
