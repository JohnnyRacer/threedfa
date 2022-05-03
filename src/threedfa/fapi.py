from importlib import reload
import uvicorn
import sys
from argparse import ArgumentParser
from app.core import config  
import os
from pathlib import Path

argspack = ArgumentParser(description='Image-fastapi startup script help',add_help=True)
argspack.add_argument('-p', '--port', type=int, default=5000, help='Select which port to run the server, defaults to 5000')
args = argspack.parse_args()

os.path.isdir(config.IMG_CACHE_DIR) or os.makedirs(config.IMG_CACHE_DIR,exist_ok=True)
os.path.isdir(config.IMG_SAVE_DIR) or os.makedirs(config.IMG_SAVE_DIR,exist_ok=True)

def main():
    uvicorn.run('app.main:app', proxy_headers=True,forwarded_allow_ips='*',port=args.port,reload=True)

if __name__ == '__main__':
   main()