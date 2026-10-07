# Terms, abbreviations and internal codes

Alphabetized by code, one entry per line. A code name does not certify the manuscript design; clinical units and recording times still require the authorized dictionary and collection-process evidence. Internal reports are not shipped; [RESULTS](RESULTS.md) and [STATUS_AND_NEXT](STATUS_AND_NEXT.md) provide self-contained summaries.

|Code|Full name or meaning|
|---|---|
|`*_history`|historical-aggregation suffix; exact windows and availability gates require derivation-code checks, see T5_FIELDS|
|`*_unknown`|unknown-code indicator suffix, not a normal value; exact rules in T5_FIELDS and derivation code|
|`a`|first-event group at POD4–31; included when cleaned-table-linked and still at risk|
|`A-D5-LGB`|availability-gated tree reference for the daily task|
|`A-formal`|machine frame identity for the Task A formal development cohort|
|`A-T4-GRU`|historical daily-task gated-recurrent sequence-selection program|
|`A-T4-TCN`|historical daily-task temporal-convolution selection program|
|`A-T4-TREE`|historical daily-task window-tree selection program|
|`A-T8-GAIN`|historical daily-task gain-based feature-selection program|
|`A1`|use-specific evaluation work item; also early measurement screening in the causal-draft context|
|`A2`|fixed-versus-daily-information comparison; also later overlap diagnostics in the causal-draft context|
|`A3`|leave-centre-out evaluation work item|
|`A7`|planning item pausing main-line architecture search|
|`abc`|historical target combining onset groups a/b/c; current formal uses a/b/none, see TASKS|
|`ACTION16`|fixed action-variable interpretation slice|
|`AGENTS`|shared-conventions file for people and coding agents|
|`AI`|artificial intelligence|
|`Aim 1`|first research aim: registry prediction and target trial emulation|
|`Aim 1b`|target-trial-emulation component of the first research aim|
|`Aim 2`|second research aim: continuous physiological signals; data not yet available|
|`AIM2_LINKAGE_NOTE`|signal-to-registry linkage note|
|`AllOperations_procloc_9`|cardiac operating-room location indicator; not a centre encoding|
|`anchor-only`|anchor control using only gestational age, diagnosis, surgery type and postoperative day|
|`AP`|average precision|
|`AP_post`|average precision on sampled retained postoperative rows|
|`AP_total`|average precision on all sampled retained rows|
|`API`|application programming interface|
|`as-of`|actually available by prediction time|
|`AUC`|area under the curve; here the receiver operating characteristic curve unless stated otherwise|
|`AUROC`|area under the receiver operating characteristic curve|
|`AUROC_post`|postoperative receiver operating characteristic area field|
|`AUROC_preop_within_phase`|AUROC within retained preoperative rows; may be undefined for a single class, not replaced with zero|
|`AUROC_total`|overall receiver operating characteristic area field|
|`B`|feature-slice context: background and clock, not all necessarily preoperative; Task B is a separate landmark prediction task|
|`b`|early postoperative event group at POD0–3; may have positive preoperative rolling labels|
|`B-only-w1`|background-and-clock-only single-day model|
|`B1`|literature-scouting work item for registry and signal linkage|
|`B2`|methods work item on mixed perioperative-phase evaluation|
|`B30`|alias for the landmark in-hospital-outcome stratification task|
|`Bd`|historical daily variant of the landmark task|
|`Bell`|clinical necrotizing-enterocolitis staging criteria; applicable version needs clinical confirmation|
|`birthwt`|birth weight; units require the authorized dictionary, see RESULTS|
|`BLOCKED`|prerequisite missing; retain the reason and report completed work|
|`Brier`|mean squared error of predicted probability against a binary outcome|
|`Brier_post_sampled`|Brier probability error on sampled retained postoperative rows|
|`Brier_total_sampled`|Brier probability error on all sampled retained rows|
|`C`|feature slice: accumulated history; logistic recipe: C is inverse regularization strength, see TEAM_TASKS|
|`c`|first event after admission but before surgery; excluded from formal|
|`C2`|historical field-timing removal-sensitivity screen; not a proven-leakage list|
|`c_aux`|auxiliary rows for the preoperative-first-event group|
|`C_h`|case-stay indicator in the local cleaned-table contract; distinct from the daily label|
|`CABSI`|catheter-associated bloodstream-infection table|
|`CACHE`|shorthand for the authorized private cache root|
|`cardsurgdtSHIFT`|source surgery-date field; date parsing and index-surgery matching in DATA_LAYOUT|
|`cardsurgdtshift_index`|clean index-surgery date serial; mapping and source-date checks in DATA_LAYOUT|
|`CAUSAL-MS-CNN`|actual learner configuration key for the CausalMS-CNN mechanism; current cards do not allocate real-data fitting|
|`CausalMS-CNN`|causal multiscale convolutional neural network|
|`CCW`|clone–censor–weight method|
|`CD3`|three-calendar-day alert cooldown rule|
|`chromsyndspecyn`|whether a chromosomal abnormality or syndrome is present; definition from authorized dictionary, codes and timing still require checks, see T5_FIELDS|
|`CI`|confidence interval|
|`ci95`|95% confidence-interval output field, see explore/README|
|`CITL`|calibration-in-the-large, usually represented by a calibration intercept|
|`CLAUDE`|entry file pointing to shared coding conventions|
|`CLI`|command-line interface|
|`CNN`|convolutional neural network|
|`CORE`|historical name for local cleaned-table row and sampling interfaces|
|`cp1252`|Windows-1252 text-encoding fallback, not a cohort version|
|`CPU`|central processing unit|
|`cross_stage_AUC`|in the EVALUATION formula, AUROC ranking postoperative positives against preoperative negatives|
|`cross_stage_positive_vs_preop_negative_auc`|general phase-diagnostic field: positive-day versus preoperative-negative ranking; see EVALUATION for the positive-phase convention|
|`CSV`|comma-separated values file|
|`cu128`|CUDA build tag of the frozen torch package; preserve the ENVIRONMENT version string|
|`CUDA`|NVIDIA's parallel-computing platform|
|`d`|first event after POD31; excluded from formal|
|`D1`|intent to isolate future new data, without carving an independent holdout from used development data|
|`D3`|convention retaining a fixed interpretation group without exposing its source learner|
|`D5`|historical reconstructed feature-bank name; not a window length|
|`D5-drop65-v1`|historical removal-sensitivity candidate and frozen-list version; drops the entire timing-uncertain set and listed derived channels, retaining original fold rounds without new tuning or model promotion; see T5_FIELDS|
|`D5-safe`|registry bank masked by the specified historical-day availability rules; safe does not establish independently verified timing for every source field|
|`D5SAFE`|configuration key for the daily availability-gated feature bank|
|`D_I_N`|field in historical action-evidence cards; letter definitions are not explained in this package and must not be inferred|
|`d_window`|within-window auxiliary rows for the late-first-event group|
|`DAG`|directed acyclic graph|
|`DATA_LAYOUT`|input, cohort and private-output layout guide|
|`DCA`|decision curve analysis|
|`DEC`|historical research-decision record prefix|
|`DEC-007`|historical evaluation-plan record; centre sensitivity is summarized in the results page|
|`DEC-008`|historical decision defining a bounded search and predeclared combinations|
|`DEC-009`|historical decision auditing the enhanced-encoding component|
|`DEC-010`|historical decision closing performance search after that search batch|
|`DEC-019`|historical shared plan to pause main-line architecture search; reasons are in the status page|
|`DEC-C2-001`|historical freeze record for the timing-uncertain field list|
|`DOA`|days since admission|
|`DOI`|digital object identifier|
|`draft_dag`|causal-graph draft interface, see explore/README|
|`DSSI`|deep surgical-site infection table|
|`DUA`|data use agreement|
|`dxg_fund_*`|derived fundamental-diagnosis group pattern; mappings require derivation code and authorized dictionary, exact names in T5_FIELDS|
|`E9-A-D5-site`|historical centre-held-out sensitivity run of the daily reference|
|`ECMO`|extracorporeal membrane oxygenation|
|`ENVIRONMENT`|environment installation and resource guide|
|`EOD`|end of day|
|`err`|erroneous-date group with event before admission; excluded|
|`ESS`|effective sample size|
|`EVALUATION`|evaluation-contract and examples document|
|`EXCLUDED`|sharing-exclusion and capability-boundary document|
|`extracardspecyn`|whether a congenital extracardiac anomaly is present; definition from authorized dictionary, codes and timing still require checks, see T5_FIELDS|
|`F`|RESULTS historical landmark ensemble: fixed weighted combination of enhanced tree G, TabPFN and a convolutional network; complete weights and settings not explained in this package; in DATA_LAYOUT rN/fF, F is only a fold-index placeholder|
|`f_pre`|fraction of all negative days that are preoperative, see EVALUATION formula|
|`FeatureRows`|input object with daily-row identities, predictors and availability metadata|
|`fit_pool_sha256`|SHA-256 digest of fitting-pool identity|
|`FitContext`|fit-context object binding training-pool identity and seeds|
|`fold_r*`|naming pattern for repeat-specific fold fields, see fold_rN|
|`fold_rN`|stay-grouped fold for repeat N; fold_r* is the corresponding naming pattern|
|`fold_site`|fixed centre group, see STATUS_AND_NEXT|
|`formal`|as a cohort code: outcome-conditioned, cleaned-table-linked development cohort; the ordinary English adjective formal does not name that cohort|
|`FPR`|false positive rate|
|`funddiagnosis_<code>`|derived fundamental-cardiac-diagnosis code pattern; exact codes in T5_FIELDS and the authorized dictionary|
|`G`|encoding context: component adding codes, support states and centre information; RESULTS historical ensemble component G is the complete tree model using these inputs and reconstructed features, not the same object|
|`G-LGB`|historical enhanced tree model without the current preoperative masking|
|`G-safe`|encoding-enhanced scheme with specified availability masking; safe does not establish independently verified timing for every source field|
|`G-safe-LGB`|documentation alias for the encoding-enhanced tree reference|
|`GAIN`|feature ranking by tree gain within the current training pool|
|`GAIN50`|top-fifty feature recipe ranked by gain within the current training pool|
|`GAIN50-w7`|training-pool gain-selected configuration with a seven-day calendar window|
|`GB`|gigabyte in decimal units|
|`gestagewks`|gestational age in weeks; background-field description in RESULTS|
|`GiB`|gibibyte in binary units|
|`GLOSSARY`|glossary of terms and internal codes|
|`GPU`|graphics processing unit|
|`GRU`|gated recurrent unit|
|`GRU-D`|gated recurrent unit with missingness decay|
|`GRUD-WINDOW`|configuration key for the missingness-decay window sequence learner|
|`GSAFE`|availability-gated encoding-enhanced variant; safe does not establish independently verified timing for every source field|
|`GSAFE-LGB`|encoding-enhanced Light Gradient Boosting Machine reference|
|`H1`|first code-handoff and equivalence-check record; summarized in the environment guide|
|`harness_v2`|historical evaluation implementation; alert conventions in STATUS_AND_NEXT|
|`harness_v3`|frozen stay-splitting and evaluation implementation version, see STATUS_AND_NEXT|
|`hospitalizationidNEW`|raw stay identifier, not a cross-stay patient identifier; see DATA_LAYOUT|
|`hospitalizationidnew`|clean stay identifier; checked against raw keys through builder mapping, see DATA_LAYOUT|
|`I7`|historical feature-slice, window, sequence and learning-curve diagnostic program|
|`ICU`|intensive care unit|
|`ID`|identifier|
|`IndexSurgHosp`|index-surgery hospitalization source table; fields and timing require the authorized dictionary|
|`INNER-THRESH`|alert rule with thresholds fixed within training and applied unchanged|
|`IntracardLine`|intracardiac-line source table; fields and timing require the authorized dictionary|
|`IRB`|institutional review board|
|`IS`|inotropic score|
|`JSON`|JavaScript Object Notation|
|`L2`|squared-norm regularization|
|`late_aux`|late-event auxiliary day set|
|`late_era`|later-data-era flag with possible follow-up truncation|
|`LCOS`|low cardiac output syndrome|
|`lcos*`|postoperative low-cardiac-output-related field pattern; measurement conditions require the authorized dictionary, exact names in T5_FIELDS|
|`LF`|line-feed character|
|`LGB`|Light Gradient Boosting Machine learner code|
|`libomp`|OpenMP numerical runtime|
|`LightGBM`|Light Gradient Boosting Machine|
|`locked`|previously evaluated temporal holdout; no longer untouched|
|`logistic.C`|inverse regularization strength in the logistic recipe, not accumulated-history slice C|
|`M1`|historical tabular and missingness-aware neural-model selection family|
|`M1-SELECT-CMSCN`|historical causal-multiscale-convolution model-selection program|
|`M1-SELECT-GRUD`|historical missingness-decay recurrent model-selection program|
|`M1-SELECT-TABM`|historical parameter-sharing tabular model-selection program|
|`M4`|Apple chip model in the recorded local environment|
|`MANIFEST`|file inventory and integrity-hash record|
|`MAR`|medication administration record|
|`max_iter`|optimizer iteration cap, see TEAM_TASKS|
|`MD`|Doctor of Medicine degree|
|`MDD`|internally minimum detectable difference; the project's prespecified AUROC promotion margin is 0.01, with complete repeats and simultaneous-interval conditions also required; not by itself a power demonstration|
|`mean_auc_loss`|mean AUROC loss after permutation, describing model reliance rather than effect, see explore/README|
|`MechVent`|mechanical-ventilation source table; fields and timing require the authorized dictionary|
|`MLP`|multilayer perceptron|
|`MRN`|medical record number|
|`NaN`|Not a Number, used as a numerical missing-value marker, not zero|
|`nan_rows`|number of numerical-missing rows, see explore/README|
|`NEC`|necrotizing enterocolitis|
|`necbelldtshift`|clean first necrotizing-enterocolitis event date serial; see DATA_LAYOUT|
|`NEW`|suffix in source identifier names; spelling does not establish a new linkage, which requires authorized mapping|
|`NICU`|neonatal intensive care unit|
|`none`|group without a known first-event date; not confirmed absence of disease|
|`NX-H4-TEAM-v1`|exploration-interface specification version identity, not this round's research status|
|`NX-H4c-v1`|previous code-handoff documentation revision|
|`NX-H6-v1`|current collaboration-language, terminology and status revision|
|`NX-T2-GRUD-GAIN50-v1`|new bounded missingness-decay modeling program identifier|
|`NX-T2-TABM-GAIN50-v1`|new bounded tabular modeling program identifier|
|`NX-T3-POD02-SEQ-v1`|early-postoperative ordering-information program; POD02 means the POD0–2 range, see TEAM_TASKS|
|`NX-T4-POD1-yB30-v1`|new postoperative-first-day landmark-stratification program identifier|
|`O/E`|observed-to-expected event ratio|
|`OLS`|ordinary least squares|
|`OOF`|out-of-fold predictions|
|`OpenMP`|shared-memory parallel programming interface; here its numerical runtime|
|`OR`|operating room|
|`ORDERLESS_MLP`|orderless multilayer-perceptron control|
|`outcome_3d`|local cleaned-table single-positive-day label; not an alias for the rolling daily label|
|`PACU`|post-anesthesia care unit|
|`PC4`|Pediatric Cardiac Critical Care Consortium|
|`PF_CACHE_ROOT`|environment variable for the authorized private cache root|
|`PF_DATA_ROOT`|environment variable for the authorized read-only data root|
|`PF_ENV`|private environment-directory variable in installation examples|
|`PF_RUN_DIR`|environment variable for the authorized private run-output directory|
|`PhDc`|Doctor of Philosophy candidate designation|
|`PI`|principal investigator|
|`PI-29`|source-model feature family excluded from the package|
|`PI10`|fixed ten-variable interpretation group; its source learner is not exposed|
|`PI29`|alias of the excluded source-model feature family|
|`PI72`|historical shorthand for the local cleaned-table contract; does not certify the manuscript design|
|`PI72-CLEAN`|local cleaned-table row, label and evaluation contract; the manuscript's final scoring design remains unresolved|
|`POD`|postoperative day|
|`POD0`|surgery day; completed-surgery information still requires recording and availability gates, not merely a nonnegative day|
|`POD02`|program-identifier range POD0–2, inclusive of surgery day through postoperative day 2; not a single day|
|`POD1`|postoperative day 1, with surgery day counted from zero|
|`POD2`|postoperative day 2, with surgery day counted from zero|
|`POD3`|postoperative day 3, with surgery day counted from zero|
|`POD31`|postoperative day 31, with surgery day counted from zero|
|`POD8`|postoperative day 8, with surgery day counted from zero|
|`POD_lead_profile`|attribution field stratified by postoperative day and event lead time|
|`positivity`|support for both treatment strategies under the specified conditions|
|`PPRL`|privacy-preserving record linkage|
|`PPV`|positive predictive value|
|`PreopRiskFactor`|preoperative-risk-factor source table; code meanings require the authorized dictionary|
|`PreopRiskFactor_330`|identifier of the removed historical preoperative-risk-factor field|
|`PreopRiskFactor_<code>`|derived preoperative-risk-factor code pattern; exact codes in T5_FIELDS and the authorized dictionary, not inferred individually|
|`ProcLoc`|operating-location field; distinct from centre encoding|
|`PYTHONDONTWRITEBYTECODE`|interpreter environment variable disabling bytecode caches|
|`PYTHONPATH`|interpreter module-search-path environment variable|
|`r0–4`|zero-based repeat identifiers for daily and landmark tasks; not new patients|
|`R1`|lag-flattened window representation; not a repeat identifier|
|`R1-LGB`|lag-flattened-window tree model|
|`R10`|historical local cleaned-table attribution report|
|`R11`|historical daily-task confirmation, attribution and closeout report|
|`r1–5`|one-based repeat identifiers for the local cleaned-table contract; distinct from the zero-based splits|
|`R3`|window representation of per-variable summaries and structural counts|
|`R4`|sequence representation with values, missingness bits and row-availability bits|
|`R8`|historical candidate-program screening report for the local cleaned-table contract|
|`R9`|RESULTS historical cleaned-table confirmation report; feature-encoding context: nine raw-code inputs, see TASKS; distinct from lowercase repeat index r|
|`rank_cards`|interface ordering evidence cards under fixed discussion rules, see explore/README|
|`rank_ci_support_ok`|whether attribution-rank interval support requirements are met, see explore/README|
|`RANK_ORDER`|fixed ordering key for causal-discussion cards|
|`RANK_REASONS`|field recording reasons for causal-discussion ordering|
|`README`|human-facing repository introduction|
|`REPRODUCE`|reproduction guide|
|`RESULTS`|aggregate-results and evidence-boundary document|
|`RETRO-CAP`|retrospective equal-capacity alert allocation using future days' scores|
|`RiskSurgVIS`|vasoactive-inotropic-score source table; snapshot anchors and availability require the authorized field contract|
|`RNG`|random number generator|
|`RSS`|resident set size|
|`RUN`|shorthand for the private run-output directory|
|`RUN_ROOT`|resolved private run-output root|
|`RX-D1`|historical evaluation-contract freeze work item|
|`RX-D1-v1`|frozen computation and seed-naming version, see RX-D1; not the current documentation version|
|`RX-H1`|first handoff-export work item|
|`RX-H1-v1`|version identity of initial-handoff provenance, see RX-H1; not the current documentation version|
|`RX-H2`|historical blind-reading and semantic-review work item|
|`RX-H3`|historical handoff-document revision work item|
|`RX-I1`|historical implementation-interface work item|
|`RX-I11`|historical fast-attribution aggregation-refactoring work item|
|`RX-I2b`|historical enhanced-encoding interface work item|
|`S`|I7: current-day state slice; S.parquet: support-state cache, see DATA_LAYOUT; the severity-proxy field is not this slice|
|`S.parquet`|support-state cache built by data.py through line_S, not the I7 current-day slice|
|`S15`|block of fifteen derived support-state features|
|`S_RISK`|severity-proxy-risk discussion field|
|`SCORER_SCHEMA`|synthetic scoring-interface schema and open-items table|
|`screen`|diagnostic screen with limited repeats; insufficient for promotion|
|`severity_proxy_S`|severity-proxy field in historical action-evidence cards; exact categories not explained in this package, distinct from current-day slice S|
|`SHA-256`|Secure Hash Algorithm file-integrity digest|
|`SHADOW`|historical shadow-variable ranking method|
|`SHAP`|SHapley Additive exPlanations; model dependence rather than treatment effect|
|`SHIFT`|date suffix in source names; spelling alone does not establish shifting rules, which require authorized mapping|
|`site1`|configuration identity of single-column centre encoding in spec_v1; see site_eb|
|`site_eb`|centre empirical-Bayes shrinkage encoding|
|`SMD`|standardized mean difference|
|`source_path`|original-project source location in MANIFEST; provenance, not an in-package runtime path|
|`splits_v3`|frozen stay-split table, see DATA_LAYOUT and TASKS|
|`SpO2`|peripheral oxygen saturation measured by pulse oximetry; measurement and quality rules await the synthetic-field contract, see SCORER_SCHEMA|
|`spo2_daily_mean`|synthetic daily-mean SpO2 field; includes only observations available by cutoff, see SCORER_SCHEMA|
|`stage3_cards`|stage 3 candidate-evidence aggregation interface, see explore/README|
|`stage4`|stage 4 trial and causal-graph drafts, not treatment-effect estimation|
|`STAGE4_IDS`|fixed set of question identifiers supported by causal drafts|
|`STAT`|Society of Thoracic Surgeons–European Association for Cardio-Thoracic Surgery congenital heart surgery risk score|
|`STATE`|internal historical state record; relevant conclusions are summarized here|
|`statscore`|surgical risk-score field, see RESULTS; exact version requires the authorized dictionary|
|`STATUS_AND_NEXT`|current-status and next-step discussion document|
|`STOP`|historical status stopped at its screening gate; neither automatic extension nor reversal of results|
|`Surgdiag`|surgical-diagnosis source table; codes and timing require the authorized dictionary|
|`T-2`|open task: bounded tabular and missingness-aware modeling|
|`T-3`|open task: early-postoperative ordering-information test|
|`T-4`|open task: postoperative-first-day landmark stratification|
|`T-5`|open task: field-availability timing audit|
|`T-6`|open task: signal interfaces and linkage governance|
|`T007`|internal manuscript-table organization record; not a shipped dependency|
|`T2`|historical fixed-prediction rescoring work item|
|`T3`|context-dependent: explore.t3 is the historical attribution module; AIM2_LINKAGE_NOTE cites an external physiological-signal platform whose full name and linkage method remain unverified, with search snippets only|
|`T3a`|historical overall-importance analysis|
|`T3b`|historical case-and-control trajectory analysis|
|`T3c`|historical postoperative-day and event-lead attribution|
|`T4`|historical window and sequence selection program|
|`T4-MATCHED-W1`|historical feature-matched single-day tree reference|
|`T4-SELECT-GRU`|historical gated-recurrent sequence-selection program|
|`T4-SELECT-TCN`|historical temporal-convolution sequence-selection program|
|`T4-SELECT-TREE`|historical window-tree selection program|
|`T5`|historical prediction-task closeout code; distinct from the hyphenated timing-audit card|
|`T5_FIELDS`|historical audit field-name list document|
|`T6`|historical action-candidate card code; distinct from the hyphenated signal-interface card|
|`T6_candidates`|aggregate-evidence field for historical action candidates|
|`T8`|historical feature-count selection program|
|`T8-D5SAFE-LGB`|full availability-gated tree reference for the local cleaned-table contract|
|`T8-D5SAFE-XGB`|historical extreme-gradient-boosted tree reference for the local cleaned-table contract|
|`T8-SELECT-LGB`|historical Light Gradient Boosting feature-count selection program|
|`T8-SELECT-XGB`|historical extreme-gradient-boosting feature-count selection program|
|`TabM`|parameter-sharing ensemble of small tabular models|
|`TABM-R1`|lag-flattened configuration key for the parameter-sharing tabular model|
|`TabPFN`|Tabular Prior-data Fitted Network; here identifying a historical ensemble component only|
|`Task A`|daily rolling prediction task|
|`Task B`|landmark risk-stratification task|
|`TASKS`|task-contract definitions document|
|`TCN`|temporal convolutional network|
|`TEAM_TASKS`|open task-card document|
|`TEAM_USAGE`|synthetic guide to single-fit and label-free application|
|`time zero`|time aligning eligibility, strategy assignment and follow-up start|
|`TREE`|tree-model program label|
|`trial_skeleton`|target-trial draft interface, see explore/README|
|`TTE`|target trial emulation|
|`U1`|research use of daily dynamic warning|
|`U2`|research use of identifying treatment-improvement space; the current step is prognostic stratification|
|`U2_CONTRACT_SUMMARY`|shared evaluation-contract summary for landmark stratification|
|`UTF-8`|variable-length Unicode text encoding|
|`UTI`|urinary-tract infection table|
|`validation_auc`|inner-validation AUROC, see TEAM_USAGE|
|`venv`|Python virtual environment|
|`VIS`|vasoactive-inotropic score|
|`w1`|single-day calendar window|
|`w3`|three-day calendar window|
|`w7`|seven-day calendar window|
|`Windows-1252`|Windows text-encoding name; see cp1252|
|`WP0`|historical local cleaned-table row and sampling reconciliation work item|
|`xclamptime`|aortic cross-clamp-time field, see RESULTS; units require the authorized dictionary|
|`XGB`|extreme gradient boosting learner code|
|`XGB-A-D5-phase_split`|historical daily-task extreme-gradient-boosting configuration split by preoperative and postoperative phase|
|`y3`|daily-task first-recorded-outcome label over the next three days; the supplementary landmark model is fitted separately|
|`y30`|alias for first in-hospital outcome in the thirty days after the landmark|
|`yB`|older label from landmark to the postoperative observation-window end; distinct from a fixed-length endpoint|
|`yB30`|first in-hospital outcome in the thirty days after a fixed prediction time|
|`z_days_since_*`|derived days-since-record pattern; anchors and masking require derivation-code checks, see T5_FIELDS|
|`z_doa`|days since admission, see RESULTS background/clock field table|
|`z_dol`|age in days, see RESULTS background/clock field table|
|`z_pod`|postoperative day, see RESULTS background/clock field table|
|`z_post_surg`|postoperative-stage indicator, see RESULTS background/clock field table|
