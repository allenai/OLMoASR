To install the OpenWhisper package: 

1. First, clone the OpenWhisper repo into your source directory
`git clone https://github.com/huongngo-8/open_whisper.git`

If you are installing packages with the requirements.txt file, you should just run after cloning the repo 
`pip install -r requirements.txt`
Otherwise, follow the steps below. 

2. After cloning the repository, change into the directory of the cloned repository:
`cd repository`

3. Now, install the package in editable mode using `pip`
`pip install -e .`

Finally, to run the data downloading process, run
`python scripts/data/data_transfer/multi_download_to_s3.py`
