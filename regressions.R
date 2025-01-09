require(emmeans)

results = read.csv("results/no_aggregation_correlation.csv")
results$reader_id <- as.factor(results$reader_id)
results$lang <- as.factor(results$lang)

load("data/joint.ind.diff.l2.rda")
load("data/joint.comp.l2.rda")

# merge participant info and comprehension data
results <- results %>%
  left_join(y = joint_id, by = join_by(reader_id==uniform_id)) %>%
  left_join(joint.comp, by = join_by(reader_id == uniform_id))

model_layer <- lm(data = results,
                  formula = "spearman_r ~ layers")
summary(model_layer)

# pairwise analysis across layers
emm <- emmeans(model_layer, "layers")
pairs(emm)

# language groups

first_layer_total <- subset(results,
                            results$measures == "Total fixation duration" &
                              results$layers == "First")

model <- lm(data = first_layer_total,
            formula = "spearman_r ~ lang.x + lextale + accuracy")
summary(model)

#compare with the grand mean
emm <- emmeans(model, "lang.x")
contrasts <- contrast(emm, "eff")
contrasts