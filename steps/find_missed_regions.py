import os
import numpy as np


nb_regions = 860

region_ids=list(range(nb_regions))
region_ids = [str(l) for l in region_ids]


pre_found_ids = os.listdir('./REGIONS_Dec23')
new_pre_found_ids = []
for id in pre_found_ids:
  f = id.find('_')
  new_pre_found_ids.append(id[f+1:-4])

#print(new_pre_found_ids)

missed_ids = list(set(region_ids) - set(new_pre_found_ids))
print(missed_ids)
print('len', len(missed_ids))


