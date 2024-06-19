import pylab
import sys
import pymzml
import numpy as np
import pickle
import os
import time
# list all files in current directory with mzML extension even in sub dir

import multiprocessing
import concurrent.futures
import pymzml
import numpy as np
from PIL import Image
import random
import os

def process_file(file):
    try:
        name = os.path.basename(file).replace(".mzML", "")
        if name+".png" in os.listdir("/srv/s01/leaves-shared/marshall/images2/png/class2"):
            return
        run = pymzml.run.Reader(file, build_index_from_scratch=True)
        TIC = []
        for i, spec in enumerate(run):
            mz = spec.peaks("centroided")
            max_mz = max(mz, key=lambda x: x[0])
            if max_mz[0] < 200:
                continue
            tmp = np.zeros(shape=(min(int(max_mz[0]), 1000) + 1))
            for m, i in mz:
                if m > 1000:
                    break
                tmp[int(m)] = i
            TIC.append(tmp)
        max_length = max([len(i) for i in TIC])
        TIC = np.array([np.pad(i, (0, max_length - len(i))) for i in TIC])
        min_mz = np.where(TIC.mean(axis=0) == 0)[0][-2]
        TIC = TIC[:, min_mz:]
        TIC2 = np.log(TIC / (TIC.max() + 1e-5) + 1e-5)
        TIC2 = (TIC2 - TIC2.min()) / (TIC2.max() - TIC2.min()) * 255
        img = Image.fromarray(TIC2).resize((2048, 2048)).convert("L")
        img.save(f'/srv/s01/leaves-shared/marshall/images2/png/class2/{name}.png')
        # np.save(f'/srv/s01/leaves-shared/marshall/images2/npy/{name}.npy', TIC2)
        print(f"Processed: {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# def main(files):
#     with concurrent.futures.ProcessPoolExecutor(max_workers=120) as executor:
#         futures = {executor.submit(process_file, file): file for file in files}
#         for future in concurrent.futures.as_completed(futures):
#             file = futures[future]
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Error processing {file}: {e}")
def main(files):
    with multiprocessing.Pool(processes=120) as pool:
        results = [pool.apply_async(process_file, (file,)) for file in files]
        
        for result, file in zip(results, files):
            try:
                result.get()
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    files = []
    for root, dirs, file in os.walk("/srv/s02/leaves-shared/marshall/"):
        for f in file:
            if f.endswith(".mzML"):
                files.append(os.path.join(root, f))

    main(files)
