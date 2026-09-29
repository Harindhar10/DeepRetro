%%writefile /content/DeepRetro/test.py
# load up USPTO-50k test dataset
import os
import pandas as pd
import json
import time
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

root_dir = os.getcwd()
from src.main import main
from src.cache import clear_cache_for_molecule

URL ="https://raw.githubusercontent.com/Sauravroy34/DeepRetro/refs/heads/main/data/USPTO_190.csv"


df = pd.read_csv(URL)
mols_dfs = df['smiles'].to_list()

folder = "USPTO"
for run_no in range(9,12):
  if not os.path.exists(f"{root_dir}/results/dfs/{folder}"):
      os.makedirs(f"{root_dir}/results/dfs/{folder}")
  if not os.path.exists(f"{root_dir}/results/dfs/{folder}/run_{run_no}"):
      os.makedirs(f"{root_dir}/results/dfs/{folder}/run_{run_no}")

  for mol in mols_dfs:
      molecule = mol
      print(f"Running {mol}")

      llm = "Qwen/Qwen2.5-0.5B-Instruct"
      az_model = "USPTO"

      time1 = time.time()
      try:
          clear_cache_for_molecule(molecule)
          print(f"Running {molecule} with {llm} and {az_model}")

          res_dict = main(molecule ,llm=llm, az_model=az_model, stability_flag="True", hallucination_check="True", local = True)

          with open(f"{root_dir}/results/dfs/{folder}/run_{run_no}/{mol}_stability_hallucination.json", "w") as f:
              json.dump(res_dict, f, indent=4)
      except Exception as e:
          print("Error in molecule:", mol)
          print("Error:", e)
      time2 = time.time()
      print(f"Time taken for {mol} on {folder}: {time2-time1} seconds")
