#!/usr/bin/env Rscript
# =============================================================================
# Level 2A -- synthetic population generation with synthpop
# =============================================================================
#
# Sequential CART synthesis: each variable is modelled on the variables already
# synthesised, which preserves the joint structure that independent marginal
# sampling would destroy.
#
# Variable ordering matters. synthpop conditions each variable on its
# predecessors in the visit sequence, so the order encodes an assumed causal
# direction: demographics first, then cancer characteristics, then
# comorbidities, then anything else. Reordering changes the synthesis.
#
# Scenario shifts are applied by replacing the age synthesis with a draw from a
# truncated normal at a shifted mean, keeping the observed standard deviation
# and clamping to the eligible age range. Everything downstream of age in the
# visit sequence is then synthesised conditional on the shifted ages, which is
# the point: an older cohort should also carry an older cohort's comorbidity
# profile, not the original one.
#
# Usage
# -----
#   Rscript R/level2_synthpop.R <input.csv> <output.csv> <n> [age_shift] [seed]
#
# Examples
# --------
#   Rscript R/level2_synthpop.R outputs/processed_synth_raw_data.csv \
#       outputs/synthpop_patients_orig_age.csv 24462 0 42
#   Rscript R/level2_synthpop.R outputs/processed_synth_raw_data.csv \
#       outputs/synthpop_patients_ageshift_plus_5.csv 24462 5 42
#
# On Windows, call the interpreter by full path, or set config.RSCRIPT and let
# run_level2.py invoke it for you:
#   "C:/Program Files/R/R-4.5.1/bin/Rscript.exe" R/level2_synthpop.R ...
#
# Requires: synthpop, truncnorm
# =============================================================================

suppressPackageStartupMessages({
  library(synthpop)
  library(truncnorm)
})

# ---------------------------------------------------------------------------
# Parameters -- edit here, not inline below
# ---------------------------------------------------------------------------

AGE_COLUMN   <- "age_at_followup_y"
AGE_MIN      <- 65   # cohort eligibility floor
AGE_MAX      <- 95   # upper truncation for the shifted distribution
MIN_LEVELS   <- 2    # synthpop minnumlevels

# Columns never synthesised: identifiers, derived quantities, and any cluster
# label (phenotype is assigned afterwards by the Python classifier, so letting
# synthpop invent one would leak a circular dependency into the pipeline).
SKIP_COLUMNS <- c(
  "patient_id", "PATIENT_ID", "comorbid_count", "latent_profile", "cluster",
  "cluster_lcs_3clus", "cluster_lcs_4clus", "cluster_pca_3clus", "cluster_pca_5clus"
)

# Visit sequence blocks, in synthesis order. Names absent from the input are
# dropped silently, so one script serves several cohort extracts.
DEMOGRAPHIC_VARS <- c(
  "age_at_followup_y", "sex_num_y", "race_grp_y",
  "YOSTACS20132017QUINTILESTATEBAS", "GRP_MARITAL_STATUS_AT_DIAGNOSIS",
  "GRP_PRIMARY_PAYER_AT_DX", "INSURANCE_RECODE_2007"
)
CANCER_VARS <- c(
  "PRIMARY_SITE", "SEERCOMBINEDSUMMARYSTAGE2000200", "GRADE_THRU_2017",
  "CHEMOTHERAPY_RECODE_YES_NO_UNK", "GRP_RADIATION_RECODE"
)
COMORBIDITY_VARS <- c(
  "AMI", "ALZH", "ATRIAL_FIB", "CATARACT", "CHRONICKIDNEY", "COPD", "CHF",
  "DIABETES", "GLAUCOMA", "HIP_FRACTURE", "ISCHEMICHEART", "DEPRESSION",
  "OSTEOPOROSIS", "RA_OA", "STROKE_TIA", "HYPERT", "HYPERL", "ANEMIA", "ASTHMA"
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(
    "usage: Rscript level2_synthpop.R <input.csv> <output.csv> <n> [age_shift] [seed]",
    call. = FALSE
  )
}

input_csv  <- args[1]
output_csv <- args[2]
n_total    <- as.integer(args[3])
age_shift  <- if (length(args) >= 4) as.numeric(args[4]) else 0
seed       <- if (length(args) >= 5) as.integer(args[5]) else 42L

if (is.na(n_total) || n_total <= 0) {
  stop(sprintf("n must be a positive integer, got '%s'", args[3]), call. = FALSE)
}
if (!file.exists(input_csv)) {
  stop(sprintf("input file not found: %s", input_csv), call. = FALSE)
}

cat(sprintf(
  "synthpop: input=%s n=%d age_shift=%+g seed=%d\n",
  basename(input_csv), n_total, age_shift, seed
))

# ---------------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------------

df <- read.csv(input_csv, stringsAsFactors = TRUE)
cat(sprintf("Loaded %d patients, %d columns\n", nrow(df), ncol(df)))

feat <- setdiff(names(df), SKIP_COLUMNS)

demo  <- intersect(DEMOGRAPHIC_VARS, feat)
cancer <- intersect(CANCER_VARS, feat)
comor <- intersect(COMORBIDITY_VARS, feat)
rest  <- setdiff(feat, c(demo, cancer, comor))
feat  <- c(demo, cancer, comor, rest)

if (length(rest) > 0) {
  cat(sprintf("Unclassified variables synthesised last: %s\n",
              paste(rest, collapse = ", ")))
}

has_age <- AGE_COLUMN %in% feat
if (!has_age && age_shift != 0) {
  stop(sprintf("age_shift requested but column '%s' is not present", AGE_COLUMN),
       call. = FALSE)
}

# Character columns must be factors for CART synthesis.
for (col in feat) {
  if (is.character(df[[col]])) df[[col]] <- as.factor(df[[col]])
}

# ---------------------------------------------------------------------------
# Custom age synthesis
# ---------------------------------------------------------------------------
# synthpop dispatches on method name: declaring method "targetage" makes it call
# syn.targetage(). Instead of learning age from the data, this draws the whole
# synthetic age vector from a truncated normal whose mean is shifted by
# `age_shift`. Because age is synthesised first, every later variable is
# conditioned on the shifted ages.

if (has_age) {
  age_sd       <- sd(df[[AGE_COLUMN]], na.rm = TRUE)
  age_mean     <- mean(df[[AGE_COLUMN]], na.rm = TRUE)
  age_syn_mean <- age_mean + age_shift

  cat(sprintf("Age: observed mean %.2f (sd %.2f) -> synthetic mean %.2f, truncated [%d, %d]\n",
              age_mean, age_sd, age_syn_mean, AGE_MIN, AGE_MAX))

  syn.targetage <- function(y, x, xp, proper = FALSE, ...) {
    new_ages <- rtruncnorm(n_total, a = AGE_MIN, b = AGE_MAX,
                           mean = age_syn_mean, sd = age_sd)
    list(res = new_ages, fit = "target age (truncated normal)")
  }
  assign("syn.targetage", syn.targetage, envir = globalenv())
}

# ---------------------------------------------------------------------------
# Synthesise
# ---------------------------------------------------------------------------
# The `dummy` column is a synthpop workaround: the first variable in a visit
# sequence has no predictors and is sampled from its own marginal. Putting a
# throwaway variable first means the first *real* variable (age) is not wasted
# in that position.

set.seed(seed)
df$dummy <- factor(sample(letters[1:3], nrow(df), replace = TRUE))

methods <- rep("cart", length(feat) + 1)
names(methods) <- c("dummy", feat)
methods["dummy"] <- "sample"
if (has_age) methods[AGE_COLUMN] <- "targetage"

visit_seq <- c("dummy", if (has_age) AGE_COLUMN, setdiff(feat, AGE_COLUMN))

synth <- syn(
  df[, c("dummy", feat), drop = FALSE],
  method         = methods,
  visit.sequence = visit_seq,
  seed           = seed,
  k              = n_total,
  print.flag     = FALSE,
  minnumlevels   = MIN_LEVELS
)

df_out <- synth$syn
df_out$dummy <- NULL

# Recompute the derived comorbidity burden rather than synthesising it, so it
# stays consistent with the synthesised flags.
if (length(comor) > 0) {
  df_out$comorbid_count <- rowSums(df_out[, comor, drop = FALSE])
}

dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(df_out, output_csv, row.names = FALSE)
cat(sprintf("Wrote %d synthetic patients to %s\n", nrow(df_out), output_csv))
