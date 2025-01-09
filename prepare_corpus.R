require(osfr)
require(tidyverse)
# eyetracking data
data <- osf_retrieve_file("https://osf.io/bf2q9") %>%
  osf_download(path = "data",
               conflicts = "overwrite")

# participant info
participants <- osf_retrieve_file("https://osf.io/bu3dz") %>%
  osf_download(path = "data",
               conflicts = "overwrite")

# comprehension data
comprehension <- osf_retrieve_file("https://osf.io/8xndq") %>%
  osf_download(path = "data",
               conflicts = "overwrite")

load(data$local_path)

dat <- joint.data[joint.data$lang != "ee", ] %>%
  group_by(itemid, sentnum) %>%
  mutate(sentid = cur_group_id()) #unique id in corpus

write_csv(dat, "data/MecoL2_11.csv")

