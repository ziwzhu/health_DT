# Cluster-stratified synthpop CART synthesis
# Usage: Rscript level2_synthpop.R input.csv output.csv n_synthetic [seed]
# "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" level2_synthpop_ageshift.R outputs/patients_clustered.csv outputs/synthpop_patients.csv 24462 #age_shift #seed
# "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" level2_synthpop_ageshift.R processed_synth_raw_data.csv synthpop_patients_ageshift_minus_5.csv 24462 -5

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]
n_total <- as.integer(args[3])
age_shift <- ifelse(length(args) >= 4, as.integer(args[4]), 0)
seed <- ifelse(length(args) >= 5, as.integer(args[5]), 42L)


# test code
# input_csv <- "new level 1 multilabel classification/processed_synth_raw_data.csv"
# output_csv <- "new level 1 multilabel classification/synthpop_patients_ageshift5.csv"
# n_total <- 24462
# seed <- 42
# age_shift = 5
  
library(synthpop)
library(truncnorm)
df <- read.csv(input_csv, stringsAsFactors = TRUE)
cat(sprintf("Loaded %d patients \n", nrow(df)))

skip <- c("patient_id", "comorbid_count", "latent_profile", "cluster_lcs_3clus", "cluster_lcs_4clus", "cluster_pca_3clus", "cluster_pca_5clus")

# define character orders, because synthpop use sequential order of previous characters to get later ones

feat <- setdiff(names(df), c(skip, "cluster"))

demo <- intersect(c("age_at_followup_y", "sex_num_y", "race_grp_y", "YOSTACS20132017QUINTILESTATEBAS", "GRP_MARITAL_STATUS_AT_DIAGNOSIS", "GRP_PRIMARY_PAYER_AT_DX", "INSURANCE_RECODE_2007"), feat)
cancer <- intersect(c("PRIMARY_SITE", "SEERCOMBINEDSUMMARYSTAGE2000200", "GRADE_THRU_2017", "CHEMOTHERAPY_RECODE_YES_NO_UNK", "GRP_RADIATION_RECODE"), feat)
comor <- intersect(c("AMI", "ALZH", "ATRIAL_FIB", "CATARACT", "CHRONICKIDNEY", "COPD", "CHF", "DIABETES", "GLAUCOMA", "HIP_FRACTURE",
                     "ISCHEMICHEART", "DEPRESSION", "OSTEOPOROSIS", "RA_OA", "STROKE_TIA", "HYPERT", "HYPERL", "ANEMIA", "ASTHMA"), feat)
rest <- setdiff(feat, c(demo, cancer, comor))
feat <- c(demo, cancer, comor, rest)

##### New function: self-defined, new age distribution #####
# check age distribution
# hist(df$age_at_followup_y)
# mean(df$age_at_followup_y)
age_sd <- sd(df$age_at_followup_y)
age_mean <- mean(df$age_at_followup_y)
age_syn_mean <- age_mean + age_shift
# add: target_age_probs definition - eg. age 65 to 95, mean = user defined, std = get from data
syn.targetage <- function(y, x, xp, proper = FALSE, ...) {
  new_ages <- rtruncnorm(n_total, a = 65, b=95, mean = age_syn_mean, sd = age_sd)
  list(res = new_ages, fit = "target age")
}

# put characteristic into factors
for (col in feat) if (is.character(df[[col]])) df[[col]] <- as.factor(df[[col]])

# start synthesizing
df$dummy <- factor(sample(letters[1:3], nrow(df), replace = TRUE))
                   
methods <- rep("cart", length(feat) + 1)
names(methods) <- c("dummy", feat)
methods["dummy"] <- "sample"
methods["age_at_followup_y"] <- "targetage"

# visit sequence: dummy first, then age, then everything else
visit_seq <- c("dummy", "age_at_followup_y", setdiff(feat, "age_at_followup_y"))

# synthesize
s <- syn(df[, c("dummy", feat), drop = FALSE],
         method = methods,
         visit.sequence = visit_seq,
         seed = seed,
         k = n_total,
         print.flag = FALSE,
         minnumlevels = 2)

# Extract synthetic data
df_s <- s$syn
df_s$dummy <- NULL
#df_s$patient_id <- seq_len(nrow(df_s))
df_s$comorbid_count <- rowSums(df_s[, comor])

#df_s <- do.call(rbind, out); df_s$patient_id <- seq_len(nrow(df_s))
write.csv(df_s, output_csv, row.names=FALSE)
cat(sprintf("Wrote %d patients to %s\n", nrow(df_s), output_csv))

##### Prev version: stratified by cluster
# clusters <- sort(unique(df$cluster))
# props <- prop.table(table(df$cluster))
# n_per <- round(props * n_total)
# n_per[length(n_per)] <- n_total - sum(n_per[-length(n_per)])
# 
# out<- list()
# for (i in seq_along(clusters)) {
#   cl <- clusters[i]; nc <- as.integer(n_per[as.character(cl)])
#   dc <- df[df$cluster == cl, feat, drop = FALSE]
#   vary <- sapply(dc, function(x) length(unique(x)) > 1)
#   s <- syn(dc[, vary, drop = FALSE], method = "cart", k = nc, seed = seed+i, print.flag = FALSE, minnumlevels = 2)
#   # check pMSE
#   # tryCatch ({
#   #   u <- suppressWarnings(utility.gen(s, dc[, vary, drop = FALSE], print.flag = FALSE))
#   #   cat(sprintf("Cluster %s: %d -> %d, pMSE = %.4f, utilVal = $.4f\n", 
#   #               cl, nrow(dc), nc, u$pMSE, u$utilVal))
#   # }, error = function(e) {
#   #   cat(sprintf("Cluster %s: %d -> %d, pMSE ratio = %s\n", cl, conditionMessage(e)))
#   # })
#   
#   #output cluster
#   sdf <- s$syn
#   for (cc in names(vary[!vary])) sdf[[cc]] <- dc[[cc]][1]
#   sdf$cluster <- cl; out[[i]] <- sdf
# }



