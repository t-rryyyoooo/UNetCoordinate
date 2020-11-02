import argparse
from functions import createParentPath
import pandas as pd
import json
from pathlib import Path

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("json")
    parser.add_argument("--log_path", default="log/log.csv")
    args = parser.parse_args()

    return args

def main(args):
    if not Path(args.log_path).exists():
        with open(args.json, "r") as f:
            json_file = json.load(f)

        df = pd.DataFrame([json_file])

    else:
        df_org = pd.read_csv(args.log_path)
        with open(args.json, "r") as f:
            json_file = json.load(f)

        df_new = pd.DataFrame([json_file])

        df = pd.concat([df_org, df_new], sort=False)
    
    createParentPath(args.log_path)
    print("Logging to {}...".format(args.log_path))
    df.to_csv(args.log_path, index=False)
    print("Done")


if __name__ == "__main__":
    args = parseArgs()
    main(args)
