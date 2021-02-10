from data_import import read_csv

# Read collected input data
samples = []
samples_sources = ['my_direct', 'my_reverse', 'udacity']
for src in samples_sources:
    samples.extend(read_csv(f"./data/{src}/driving_log.csv", speed_limit=0.1))
