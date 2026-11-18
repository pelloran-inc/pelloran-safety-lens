#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate 200 matched-shape medical A/B/C triples and merge them with
the 100 base triples (data/traps/medical_base_100.json) into the
300-triple set. Curation follows (curate_medical_contra_flip.py).

Invariants (paper §3.1):
  - Every A, B, C is 10-30 words (matched-shape)
  - B differs from A by ONE meaningful term that reverses meaning
  - C paraphrases A (same intent, different surface wording)
  - No two triples share > 50% of clause_a tokens
  - No clause_a in this batch matches any clause_a in the base set

Trap types (50 each = 200 total):
  dosage_flip           : numeric dose off by an order of magnitude, wrong route,
                          or wrong frequency
  contraindication_flip : "indicated in" <-> "contraindicated in" flip on drug X
                          in scenario Y (drug-population mismatch)
  population_flip       : correct drug + wrong population (pediatric/pregnant/
                          renal-impaired/geriatric/lactating)
  drug_substitution     : correct drug swapped with a Look-Alike-Sound-Alike drug
                          from a different pharmacologic class
"""
import json
import hashlib
import sys
from pathlib import Path


def check_distinct(cases, existing_a_texts):
    """Verify every clause_a is unique across (new-batch, existing-100)."""
    seen = set()
    for c in cases:
        a_key = c["clause_a"].strip().lower()
        if a_key in seen:
            raise ValueError(f"duplicate clause_a within batch: {c['clause_a']}")
        if a_key in existing_a_texts:
            raise ValueError(f"clause_a collides with existing: {c['clause_a']}")
        seen.add(a_key)
    return True


def word_count(text):
    return len(text.split())


def check_shape(cases, wmin=5, wmax=30):
    """All A, B, C in wmin-wmax words, matched-shape (within 30% length).

    Collects ALL violations across cases and reports them together so we can
    fix in one pass. Returns True if clean, else raises with the full list.
    """
    violations = []
    for i, c in enumerate(cases):
        for field in ("clause_a", "clause_b", "clause_c"):
            n = word_count(c[field])
            if n < wmin or n > wmax:
                violations.append(
                    f"  case {i} ({c['trap_type']}) {field}={n} words: {c[field][:70]}..."
                )
        la, lb, lc = word_count(c["clause_a"]), word_count(c["clause_b"]), word_count(c["clause_c"])
        if max(la, lb, lc) > 2.0 * min(la, lb, lc):
            violations.append(
                f"  case {i} ({c['trap_type']}) shape a={la} b={lb} c={lc}: {c['clause_c'][:70]}..."
            )
    if violations:
        raise ValueError(
            f"{len(violations)} shape violations:\n" + "\n".join(violations)
        )
    return True


# ============================================================
# DOSAGE_FLIP — 50 triples (drug + dose numeric flip, wrong-by-order)
# ============================================================

DOSAGE_TRIPLES = [
    # 1
    ("Rosuvastatin 20 mg once daily is prescribed for adults with hyperlipidemia.",
     "Rosuvastatin 200 mg once daily is prescribed for adults with hyperlipidemia.",
     "For adult hyperlipidemia, rosuvastatin 20 mg QD is a first-line agent."),
    # 2
    ("Lisinopril 10 mg once daily controls essential hypertension in adults.",
     "Lisinopril 1000 mg once daily controls essential hypertension in adults.",
     "Adults with essential hypertension receive lisinopril 10 mg QD."),
    # 3
    ("Losartan 50 mg once daily is used for adults with primary hypertension.",
     "Losartan 5 mg once daily is used for adults with primary hypertension.",
     "For adult primary hypertension, prescribe losartan 50 mg once daily."),
    # 4
    ("Amlodipine 5 mg once daily is standard for adults with mild hypertension.",
     "Amlodipine 500 mg once daily is standard for adults with mild hypertension.",
     "Adults with mild hypertension take amlodipine 5 mg QD."),
    # 5
    ("Furosemide 40 mg twice daily is prescribed for adult volume overload in heart failure.",
     "Furosemide 4 mg twice daily is prescribed for adult volume overload in heart failure.",
     "Adult heart-failure edema is treated with furosemide 40 mg BID."),
    # 6
    ("Hydrochlorothiazide 25 mg once daily is used for adult essential hypertension.",
     "Hydrochlorothiazide 2500 mg once daily is used for adult essential hypertension.",
     "Adults with essential hypertension take HCTZ 25 mg QD."),
    # 7
    ("Spironolactone 25 mg once daily is added for adult heart failure with reduced ejection fraction.",
     "Spironolactone 2500 mg once daily is added for adult heart failure with reduced ejection fraction.",
     "For adult HFrEF, add spironolactone 25 mg QD."),
    # 8
    ("Simvastatin 40 mg once daily at bedtime treats adult hypercholesterolemia.",
     "Simvastatin 4 mg once daily at bedtime treats adult hypercholesterolemia.",
     "Adults with hypercholesterolemia receive simvastatin 40 mg QHS."),
    # 9
    ("Levothyroxine 100 mcg once daily is standard replacement for adult primary hypothyroidism.",
     "Levothyroxine 1000 mcg once daily is standard replacement for adult primary hypothyroidism.",
     "Adults with primary hypothyroidism take levothyroxine 100 mcg QD."),
    # 10
    ("Prednisone 60 mg once daily is used for adult acute severe asthma exacerbation.",
     "Prednisone 6 mg once daily is used for adult acute severe asthma exacerbation.",
     "For adult severe asthma exacerbation, give prednisone 60 mg QD."),
    # 11
    ("Omeprazole 20 mg once daily is prescribed for adult gastroesophageal reflux disease.",
     "Omeprazole 2000 mg once daily is prescribed for adult gastroesophageal reflux disease.",
     "Adults with GERD take omeprazole 20 mg QD."),
    # 12
    ("Sertraline 50 mg once daily is initiated for adult major depressive disorder.",
     "Sertraline 5 mg once daily is initiated for adult major depressive disorder.",
     "Adults with MDD start on sertraline 50 mg QD."),
    # 13
    ("Fluoxetine 20 mg once daily is a first-line agent for adult depression.",
     "Fluoxetine 2 mg once daily is a first-line agent for adult depression.",
     "For adult depression, fluoxetine 20 mg QD is first-line."),
    # 14
    ("Escitalopram 10 mg once daily is prescribed for adult generalized anxiety disorder.",
     "Escitalopram 100 mg once daily is prescribed for adult generalized anxiety disorder.",
     "Adults with GAD take escitalopram 10 mg QD."),
    # 15
    ("Alprazolam 0.5 mg three times daily is used for adult panic disorder.",
     "Alprazolam 50 mg three times daily is used for adult panic disorder.",
     "For adult panic disorder, alprazolam 0.5 mg TID is standard."),
    # 16
    ("Lorazepam 1 mg every eight hours treats adult acute anxiety.",
     "Lorazepam 10 mg every eight hours treats adult acute anxiety.",
     "Adults with acute anxiety receive lorazepam 1 mg q8h."),
    # 17
    ("Insulin glargine 20 units subcutaneous once daily is used for adult type-1 diabetes basal coverage.",
     "Insulin glargine 200 units subcutaneous once daily is used for adult type-1 diabetes basal coverage.",
     "Adult T1DM basal coverage uses insulin glargine 20 units SC QD."),
    # 18
    ("Enoxaparin 40 mg subcutaneous once daily is prescribed for adult DVT prophylaxis.",
     "Enoxaparin 4 mg subcutaneous once daily is prescribed for adult DVT prophylaxis.",
     "For adult DVT prophylaxis, use enoxaparin 40 mg SC QD."),
    # 19
    ("Ciprofloxacin 500 mg twice daily is used for adult uncomplicated urinary tract infection.",
     "Ciprofloxacin 5 mg twice daily is used for adult uncomplicated urinary tract infection.",
     "Adult uncomplicated UTI is treated with ciprofloxacin 500 mg BID."),
    # 20
    ("Azithromycin 500 mg once daily on day one, then 250 mg daily, is used for adult community-acquired pneumonia.",
     "Azithromycin 5 mg once daily on day one, then 2.5 mg daily, is used for adult community-acquired pneumonia.",
     "Adult CAP receives azithromycin 500 mg day one then 250 mg QD."),
    # 21
    ("Doxycycline 100 mg twice daily treats adult early Lyme disease.",
     "Doxycycline 1000 mg twice daily treats adult early Lyme disease.",
     "Adults with early Lyme disease take doxycycline 100 mg BID."),
    # 22
    ("Cephalexin 500 mg every six hours is used for adult uncomplicated cellulitis.",
     "Cephalexin 5 mg every six hours is used for adult uncomplicated cellulitis.",
     "Adult uncomplicated cellulitis is treated with cephalexin 500 mg q6h."),
    # 23
    ("Acetaminophen 1000 mg every six hours treats adult mild to moderate pain.",
     "Acetaminophen 10000 mg every six hours treats adult mild to moderate pain.",
     "For adult mild pain, give acetaminophen 1000 mg q6h."),
    # 24
    ("Naproxen 500 mg twice daily is used for adult rheumatoid arthritis flare.",
     "Naproxen 5 mg twice daily is used for adult rheumatoid arthritis flare.",
     "Adults with RA flare take naproxen 500 mg BID."),
    # 25
    ("Sildenafil 50 mg once daily is prescribed for adult erectile dysfunction.",
     "Sildenafil 500 mg once daily is prescribed for adult erectile dysfunction.",
     "For adult erectile dysfunction, sildenafil 50 mg QD is standard."),
    # 26
    ("Tamsulosin 0.4 mg once daily is used for adult benign prostatic hyperplasia symptoms.",
     "Tamsulosin 4 mg once daily is used for adult benign prostatic hyperplasia symptoms.",
     "Adult BPH is treated with tamsulosin 0.4 mg QD."),
    # 27
    ("Ondansetron 4 mg every eight hours is prescribed for adult chemotherapy-induced nausea.",
     "Ondansetron 400 mg every eight hours is prescribed for adult chemotherapy-induced nausea.",
     "For adult chemo-induced nausea, ondansetron 4 mg q8h is standard."),
    # 28
    ("Salbutamol 200 mcg inhaled every four hours as needed is used for adult acute asthma.",
     "Salbutamol 2000 mcg inhaled every four hours as needed is used for adult acute asthma.",
     "Adult acute asthma is treated with salbutamol 200 mcg inhaled q4h PRN."),
    # 29
    ("Budesonide 400 mcg inhaled twice daily is used for adult asthma maintenance.",
     "Budesonide 4 mcg inhaled twice daily is used for adult asthma maintenance.",
     "For adult asthma maintenance, budesonide 400 mcg BID is standard."),
    # 30
    ("Montelukast 10 mg once daily in the evening is prescribed for adult asthma prophylaxis.",
     "Montelukast 100 mg once daily in the evening is prescribed for adult asthma prophylaxis.",
     "Adults on asthma prophylaxis take montelukast 10 mg QPM."),
    # 31
    ("Digoxin 0.25 mg once daily is used for adult heart failure with reduced ejection fraction.",
     "Digoxin 25 mg once daily is used for adult heart failure with reduced ejection fraction.",
     "For adult HFrEF, digoxin 0.25 mg QD is a maintenance dose."),
    # 32
    ("Dabigatran 150 mg twice daily is prescribed for adult atrial fibrillation stroke prevention.",
     "Dabigatran 15 mg twice daily is prescribed for adult atrial fibrillation stroke prevention.",
     "Adult AFib stroke prevention uses dabigatran 150 mg BID."),
    # 33
    ("Fluconazole 150 mg as a single oral dose treats adult uncomplicated vulvovaginal candidiasis.",
     "Fluconazole 1500 mg as a single oral dose treats adult uncomplicated vulvovaginal candidiasis.",
     "Adult uncomplicated vaginal candidiasis is treated with fluconazole 150 mg PO once."),
    # 34
    ("Acyclovir 400 mg five times daily treats adult acute herpes simplex outbreak.",
     "Acyclovir 4 mg five times daily treats adult acute herpes simplex outbreak.",
     "For adult acute HSV outbreak, acyclovir 400 mg five times daily is standard."),
    # 35
    ("Methotrexate 15 mg orally once weekly is used for adult rheumatoid arthritis.",
     "Methotrexate 15 mg orally once daily is used for adult rheumatoid arthritis.",
     "Adults with RA take methotrexate 15 mg PO once weekly."),
    # 36
    ("Apixaban 5 mg twice daily is prescribed for adult atrial fibrillation stroke prevention.",
     "Apixaban 50 mg twice daily is prescribed for adult atrial fibrillation stroke prevention.",
     "Adult AFib stroke prevention uses apixaban 5 mg BID."),
    # 37
    ("Rivaroxaban 20 mg once daily with the evening meal is used for adult AFib.",
     "Rivaroxaban 200 mg once daily with the evening meal is used for adult AFib.",
     "Adults with AFib take rivaroxaban 20 mg QD with dinner."),
    # 38
    ("Clopidogrel 75 mg once daily is used for adult secondary prevention after myocardial infarction.",
     "Clopidogrel 750 mg once daily is used for adult secondary prevention after myocardial infarction.",
     "Post-MI adults take clopidogrel 75 mg QD for secondary prevention."),
    # 39
    ("Ticagrelor 90 mg twice daily is prescribed for adult acute coronary syndrome.",
     "Ticagrelor 9 mg twice daily is prescribed for adult acute coronary syndrome.",
     "Adult ACS is treated with ticagrelor 90 mg BID."),
    # 40
    ("Levetiracetam 500 mg twice daily is used for adult focal seizure control.",
     "Levetiracetam 5 mg twice daily is used for adult focal seizure control.",
     "For adult focal seizures, levetiracetam 500 mg BID is standard."),
    # 41
    ("Phenytoin 100 mg three times daily is used for adult generalized tonic-clonic seizure maintenance.",
     "Phenytoin 1000 mg three times daily is used for adult generalized tonic-clonic seizure maintenance.",
     "Adult tonic-clonic seizure maintenance uses phenytoin 100 mg TID."),
    # 42
    ("Carbamazepine 200 mg twice daily is used for adult trigeminal neuralgia.",
     "Carbamazepine 2000 mg twice daily is used for adult trigeminal neuralgia.",
     "For adult trigeminal neuralgia, carbamazepine 200 mg BID is first-line."),
    # 43
    ("Valproate 500 mg twice daily is prescribed for adult acute mania in bipolar disorder.",
     "Valproate 5000 mg twice daily is prescribed for adult acute mania in bipolar disorder.",
     "Adult bipolar acute mania is treated with valproate 500 mg BID."),
    # 44
    ("Lithium 300 mg three times daily is used for adult bipolar disorder maintenance.",
     "Lithium 3000 mg three times daily is used for adult bipolar disorder maintenance.",
     "Adult bipolar maintenance uses lithium 300 mg TID."),
    # 45
    ("Quetiapine 100 mg twice daily is used for adult bipolar depression.",
     "Quetiapine 10 mg twice daily is used for adult bipolar depression.",
     "For adult bipolar depression, quetiapine 100 mg BID is a treatment option."),
    # 46
    ("Risperidone 2 mg twice daily is used for adult schizophrenia maintenance.",
     "Risperidone 20 mg twice daily is used for adult schizophrenia maintenance.",
     "Adult schizophrenia maintenance uses risperidone 2 mg BID."),
    # 47
    ("Haloperidol 5 mg intramuscular every four hours as needed treats adult acute agitation.",
     "Haloperidol 50 mg intramuscular every four hours as needed treats adult acute agitation.",
     "For adult acute agitation, haloperidol 5 mg IM q4h PRN is standard."),
    # 48
    ("Diphenhydramine 25 mg every six hours as needed is used for adult acute urticaria.",
     "Diphenhydramine 250 mg every six hours as needed is used for adult acute urticaria.",
     "Adult acute urticaria is treated with diphenhydramine 25 mg q6h PRN."),
    # 49
    ("Metoprolol 25 mg twice daily is used for adult heart failure with reduced ejection fraction.",
     "Metoprolol 250 mg twice daily is used for adult heart failure with reduced ejection fraction.",
     "Adults with HFrEF take metoprolol 25 mg BID."),
    # 50
    ("Propranolol 40 mg twice daily is used for adult essential tremor.",
     "Propranolol 4 mg twice daily is used for adult essential tremor.",
     "Adult essential tremor is treated with propranolol 40 mg BID."),
]


# ============================================================
# CONTRAINDICATION_FLIP — 50 triples ("indicated" <-> "contraindicated" flip)
# ============================================================

CONTRAINDICATION_TRIPLES = [
    # 1
    ("NSAIDs are contraindicated in adults with active peptic ulcer disease.",
     "NSAIDs are indicated in adults with active peptic ulcer disease.",
     "Adults with active PUD should not take NSAIDs."),
    # 2
    ("Non-selective beta-blockers are contraindicated in patients with severe bronchial asthma.",
     "Non-selective beta-blockers are indicated in patients with severe bronchial asthma.",
     "Severe asthma is a contraindication for non-selective beta-blockade."),
    # 3
    ("Metformin is contraindicated in adults with eGFR below 30 milliliters per minute.",
     "Metformin is indicated in adults with eGFR below 30 milliliters per minute.",
     "Adults with eGFR under 30 should not receive metformin."),
    # 4
    ("ACE inhibitors are contraindicated in patients with known bilateral renal artery stenosis.",
     "ACE inhibitors are indicated in patients with known bilateral renal artery stenosis.",
     "Bilateral renal artery stenosis rules out ACE-inhibitor use."),
    # 5
    ("Spironolactone is contraindicated in patients with serum potassium above 5.5 milliequivalents per liter.",
     "Spironolactone is indicated in patients with serum potassium above 5.5 milliequivalents per liter.",
     "Serum K+ above 5.5 mEq/L contraindicates spironolactone."),
    # 6
    ("Statins are contraindicated in patients with active liver disease and elevated transaminases.",
     "Statins are indicated in patients with active liver disease and elevated transaminases.",
     "Active liver disease with elevated LFTs is a contraindication for statins."),
    # 7
    ("Fluoroquinolones are contraindicated in pediatric patients under 18 years for routine infections.",
     "Fluoroquinolones are indicated in pediatric patients under 18 years for routine infections.",
     "Routine pediatric infections should not be treated with fluoroquinolones."),
    # 8
    ("Tetracyclines are contraindicated in pregnancy due to fetal tooth staining.",
     "Tetracyclines are indicated in pregnancy due to fetal tooth staining.",
     "Pregnancy contraindicates tetracycline use because of fetal tooth staining."),
    # 9
    ("Nitroglycerin is contraindicated in patients who took a PDE5 inhibitor in the last 24 hours.",
     "Nitroglycerin is indicated in patients who took a PDE5 inhibitor in the last 24 hours.",
     "Recent PDE5 inhibitor use rules out nitroglycerin administration."),
    # 10
    ("Metformin is contraindicated in patients receiving iodinated contrast with impaired renal function.",
     "Metformin is indicated in patients receiving iodinated contrast with impaired renal function.",
     "Iodinated contrast plus impaired renal function is a contraindication for metformin."),
    # 11
    ("Warfarin is contraindicated in patients with active major bleeding.",
     "Warfarin is indicated in patients with active major bleeding.",
     "Active major bleeding rules out warfarin therapy."),
    # 12
    ("Digoxin is contraindicated in patients with active ventricular fibrillation.",
     "Digoxin is indicated in patients with active ventricular fibrillation.",
     "Active ventricular fibrillation is a contraindication for digoxin."),
    # 13
    ("Beta-blockers are contraindicated in patients with acute cocaine intoxication and hypertension.",
     "Beta-blockers are indicated in patients with acute cocaine intoxication and hypertension.",
     "Cocaine intoxication with hypertension rules out beta-blocker use."),
    # 14
    ("Amiodarone is contraindicated in patients with baseline QT prolongation over 500 milliseconds.",
     "Amiodarone is indicated in patients with baseline QT prolongation over 500 milliseconds.",
     "QT prolongation above 500 ms is a contraindication for amiodarone."),
    # 15
    ("Sildenafil is contraindicated in patients taking any form of nitrates.",
     "Sildenafil is indicated in patients taking any form of nitrates.",
     "Concurrent nitrate use rules out sildenafil administration."),
    # 16
    ("Sotalol is contraindicated in patients with baseline QT interval over 500 milliseconds.",
     "Sotalol is indicated in patients with baseline QT interval over 500 milliseconds.",
     "Baseline QT over 500 ms contraindicates sotalol therapy."),
    # 17
    ("ACE inhibitors are contraindicated in pregnancy due to fetal renal toxicity.",
     "ACE inhibitors are indicated in pregnancy due to fetal renal toxicity.",
     "Pregnancy is an absolute contraindication for ACE-inhibitor use."),
    # 18
    ("Angiotensin receptor blockers are contraindicated in pregnancy.",
     "Angiotensin receptor blockers are indicated in pregnancy.",
     "Pregnancy rules out ARB therapy due to teratogenic risk."),
    # 19
    ("Cephalosporins are contraindicated in patients with prior anaphylaxis to penicillin.",
     "Cephalosporins are indicated in patients with prior anaphylaxis to penicillin.",
     "Prior anaphylaxis to penicillin rules out cephalosporin therapy."),
    # 20
    ("Sulfamethoxazole-trimethoprim is contraindicated in patients with G6PD deficiency.",
     "Sulfamethoxazole-trimethoprim is indicated in patients with G6PD deficiency.",
     "G6PD deficiency contraindicates TMP-SMX because of hemolysis risk."),
    # 21
    ("Sulfonylureas are contraindicated in patients with severe renal impairment.",
     "Sulfonylureas are indicated in patients with severe renal impairment.",
     "Severe renal impairment rules out sulfonylurea therapy."),
    # 22
    ("GLP-1 receptor agonists are contraindicated in patients with medullary thyroid carcinoma.",
     "GLP-1 receptor agonists are indicated in patients with medullary thyroid carcinoma.",
     "Medullary thyroid carcinoma is a contraindication for GLP-1 agonists."),
    # 23
    ("Fluoxetine is contraindicated within 14 days of stopping a monoamine oxidase inhibitor.",
     "Fluoxetine is indicated within 14 days of stopping a monoamine oxidase inhibitor.",
     "Recent MAOI use within 14 days rules out starting fluoxetine."),
    # 24
    ("Tricyclic antidepressants are contraindicated in the acute post-myocardial-infarction period.",
     "Tricyclic antidepressants are indicated in the acute post-myocardial-infarction period.",
     "Acute post-MI period rules out TCA therapy."),
    # 25
    ("Bupropion is contraindicated in patients with a known seizure disorder.",
     "Bupropion is indicated in patients with a known seizure disorder.",
     "Known seizure disorder is a contraindication for bupropion."),
    # 26
    ("Live viral vaccines are contraindicated in patients with severe immunosuppression.",
     "Live viral vaccines are indicated in patients with severe immunosuppression.",
     "Severe immunosuppression rules out live viral vaccination."),
    # 27
    ("Methotrexate is contraindicated in pregnancy due to teratogenicity.",
     "Methotrexate is indicated in pregnancy due to teratogenicity.",
     "Pregnancy contraindicates methotrexate because of teratogenic effects."),
    # 28
    ("Combined oral contraceptives are contraindicated in patients with active venous thromboembolism.",
     "Combined oral contraceptives are indicated in patients with active venous thromboembolism.",
     "Active VTE rules out combined oral contraceptive use."),
    # 29
    ("Aspirin is contraindicated in children with viral febrile illness due to Reye syndrome risk.",
     "Aspirin is indicated in children with viral febrile illness due to Reye syndrome risk.",
     "Viral fever in children rules out aspirin because of Reye syndrome."),
    # 30
    ("Bisphosphonates are contraindicated in patients with active esophageal stricture.",
     "Bisphosphonates are indicated in patients with active esophageal stricture.",
     "Active esophageal stricture rules out bisphosphonate therapy."),
    # 31
    ("Beta-blockers are contraindicated in decompensated acute heart failure with pulmonary edema.",
     "Beta-blockers are indicated in decompensated acute heart failure with pulmonary edema.",
     "Decompensated acute heart failure rules out beta-blocker initiation."),
    # 32
    ("Systemic corticosteroids are contraindicated in patients with active systemic fungal infection.",
     "Systemic corticosteroids are indicated in patients with active systemic fungal infection.",
     "Active systemic fungal infection is a contraindication for steroids."),
    # 33
    ("Enoxaparin is contraindicated in patients with heparin-induced thrombocytopenia.",
     "Enoxaparin is indicated in patients with heparin-induced thrombocytopenia.",
     "HIT rules out enoxaparin therapy."),
    # 34
    ("NSAIDs are contraindicated in the third trimester of pregnancy.",
     "NSAIDs are indicated in the third trimester of pregnancy.",
     "Third-trimester pregnancy rules out NSAID use."),
    # 35
    ("Aspirin is contraindicated in patients with severe hemophilia.",
     "Aspirin is indicated in patients with severe hemophilia.",
     "Severe hemophilia is a contraindication for aspirin therapy."),
    # 36
    ("Metoclopramide is contraindicated in patients with confirmed mechanical bowel obstruction.",
     "Metoclopramide is indicated in patients with confirmed mechanical bowel obstruction.",
     "Confirmed mechanical bowel obstruction rules out metoclopramide."),
    # 37
    ("Loperamide is contraindicated in patients with bloody diarrhea and suspected invasive colitis.",
     "Loperamide is indicated in patients with bloody diarrhea and suspected invasive colitis.",
     "Bloody diarrhea with suspected invasive colitis rules out loperamide."),
    # 38
    ("Codeine is contraindicated in ultra-rapid CYP2D6 metabolizers who are breastfeeding.",
     "Codeine is indicated in ultra-rapid CYP2D6 metabolizers who are breastfeeding.",
     "Ultra-rapid CYP2D6 metabolism in breastfeeding rules out codeine."),
    # 39
    ("Immediate-release nifedipine is contraindicated in acute hypertensive emergency.",
     "Immediate-release nifedipine is indicated in acute hypertensive emergency.",
     "Acute hypertensive emergency rules out immediate-release nifedipine."),
    # 40
    ("NSAIDs are contraindicated in patients with decompensated heart failure.",
     "NSAIDs are indicated in patients with decompensated heart failure.",
     "Decompensated heart failure is a contraindication for NSAID use."),
    # 41
    ("Beta-2 agonists are contraindicated in patients with severe uncorrected hypokalemia.",
     "Beta-2 agonists are indicated in patients with severe uncorrected hypokalemia.",
     "Severe uncorrected hypokalemia rules out beta-2 agonist therapy."),
    # 42
    ("Warfarin is contraindicated during pregnancy due to warfarin embryopathy.",
     "Warfarin is indicated during pregnancy due to warfarin embryopathy.",
     "Pregnancy rules out warfarin because of embryopathic effects."),
    # 43
    ("Rituximab is contraindicated in patients with active untreated hepatitis B infection.",
     "Rituximab is indicated in patients with active untreated hepatitis B infection.",
     "Active untreated hepatitis B rules out rituximab therapy."),
    # 44
    ("Isoniazid is contraindicated in patients with acute drug-induced hepatitis.",
     "Isoniazid is indicated in patients with acute drug-induced hepatitis.",
     "Acute drug-induced hepatitis rules out isoniazid initiation."),
    # 45
    ("Nitrofurantoin is contraindicated in adults with creatinine clearance below 30 milliliters per minute.",
     "Nitrofurantoin is indicated in adults with creatinine clearance below 30 milliliters per minute.",
     "CrCl below 30 mL/min rules out nitrofurantoin use."),
    # 46
    ("Ergotamines are contraindicated in patients with poorly controlled hypertension.",
     "Ergotamines are indicated in patients with poorly controlled hypertension.",
     "Poorly controlled hypertension rules out ergotamine therapy."),
    # 47
    ("Triptans are contraindicated in patients with a history of ischemic stroke.",
     "Triptans are indicated in patients with a history of ischemic stroke.",
     "Prior ischemic stroke rules out triptan use for migraine."),
    # 48
    ("Metformin is contraindicated in patients with active severe lactic acidosis.",
     "Metformin is indicated in patients with active severe lactic acidosis.",
     "Active severe lactic acidosis is a contraindication for metformin."),
    # 49
    ("Fluoroquinolones are contraindicated in patients with active tendon rupture history.",
     "Fluoroquinolones are indicated in patients with active tendon rupture history.",
     "Prior tendon rupture history rules out fluoroquinolone use."),
    # 50
    ("Live rotavirus vaccine is contraindicated in infants with severe combined immunodeficiency.",
     "Live rotavirus vaccine is indicated in infants with severe combined immunodeficiency.",
     "Severe combined immunodeficiency rules out live rotavirus vaccination."),
]


# ============================================================
# POPULATION_FLIP — 50 triples (right drug, wrong population)
# ============================================================

POPULATION_TRIPLES = [
    # 1
    ("Aspirin is used for secondary cardiovascular prevention in adults over 50.",
     "Aspirin is used for secondary cardiovascular prevention in children over 50.",
     "Adults over 50 receive aspirin for secondary cardiovascular prevention."),
    # 2
    ("NSAIDs are avoided in third-trimester pregnancy due to premature ductus closure.",
     "NSAIDs are avoided in third-trimester adolescence due to premature ductus closure.",
     "Third-trimester pregnancy is a period to avoid NSAIDs because of ductus closure."),
    # 3
    ("ACE inhibitors are avoided in pregnant patients due to teratogenicity.",
     "ACE inhibitors are avoided in male patients due to teratogenicity.",
     "Pregnant patients should not receive ACE inhibitors due to fetal risk."),
    # 4
    ("Tetracyclines are avoided in children under 8 years old due to tooth staining.",
     "Tetracyclines are avoided in adults under 8 kg due to tooth staining.",
     "Children under 8 years should not receive tetracyclines because of dental staining."),
    # 5
    ("Fluoroquinolones are avoided in growing children due to cartilage toxicity.",
     "Fluoroquinolones are avoided in growing adults due to cartilage toxicity.",
     "Growing children should not receive fluoroquinolones because of cartilage effects."),
    # 6
    ("Warfarin is not used in pregnant patients due to warfarin embryopathy.",
     "Warfarin is not used in obese patients due to warfarin embryopathy.",
     "Pregnant patients avoid warfarin because of embryopathy."),
    # 7
    ("Statins are avoided in pregnant patients due to potential fetal harm.",
     "Statins are avoided in geriatric patients due to potential fetal harm.",
     "Pregnancy is a period to avoid statin therapy."),
    # 8
    ("Metformin dose is reduced in adults with eGFR below 45 milliliters per minute.",
     "Metformin dose is reduced in children with eGFR below 45 milliliters per minute.",
     "Adults with eGFR under 45 require reduced metformin dosing."),
    # 9
    ("Digoxin dose is reduced in patients with severe renal impairment.",
     "Digoxin dose is reduced in patients with severe hepatic impairment.",
     "Severe renal impairment requires digoxin dose reduction."),
    # 10
    ("Vancomycin dosing is renally adjusted in patients with reduced creatinine clearance.",
     "Vancomycin dosing is hepatically adjusted in patients with reduced creatinine clearance.",
     "Reduced creatinine clearance mandates renal adjustment of vancomycin."),
    # 11
    ("Insulin therapy is essential in type-1 diabetes for basal glycemic control.",
     "Insulin therapy is essential in type-2 diabetes for basal glycemic control.",
     "Type-1 diabetes requires insulin for basal glycemic control."),
    # 12
    ("Live viral vaccines are avoided in severely immunocompromised patients.",
     "Live viral vaccines are avoided in severely healthy patients.",
     "Severe immunocompromise rules out live viral vaccination."),
    # 13
    ("Metoclopramide is avoided in geriatric patients due to tardive dyskinesia risk.",
     "Metoclopramide is avoided in pediatric patients due to tardive dyskinesia risk.",
     "Geriatric patients should not receive chronic metoclopramide due to tardive dyskinesia."),
    # 14
    ("Diphenhydramine is avoided in geriatric patients due to strong anticholinergic effects.",
     "Diphenhydramine is avoided in pediatric patients due to strong anticholinergic effects.",
     "Geriatric patients should avoid diphenhydramine because of anticholinergic burden."),
    # 15
    ("Long-acting benzodiazepines are avoided in geriatric patients due to fall risk.",
     "Long-acting benzodiazepines are avoided in adolescent patients due to fall risk.",
     "Geriatric patients should avoid long-acting benzodiazepines because of fall risk."),
    # 16
    ("Anticholinergic drugs are avoided in geriatric patients due to cognitive impairment.",
     "Anticholinergic drugs are avoided in neonatal patients due to cognitive impairment.",
     "Geriatric patients should avoid anticholinergic burden."),
    # 17
    ("Codeine is avoided in breastfeeding mothers who are ultra-rapid CYP2D6 metabolizers.",
     "Codeine is avoided in nonbreastfeeding mothers who are ultra-rapid CYP2D6 metabolizers.",
     "Breastfeeding mothers with ultra-rapid CYP2D6 should not receive codeine."),
    # 18
    ("Amiodarone dosing considers thyroid function and requires baseline TSH.",
     "Amiodarone dosing considers renal function and requires baseline TSH.",
     "Thyroid function status guides amiodarone monitoring."),
    # 19
    ("Acetaminophen maximum dose is reduced in patients with chronic hepatic impairment.",
     "Acetaminophen maximum dose is reduced in patients with chronic pulmonary impairment.",
     "Chronic hepatic impairment mandates reduced acetaminophen dosing."),
    # 20
    ("Opioid doses require reduction in patients with severe renal impairment.",
     "Opioid doses require reduction in patients with severe pulmonary impairment.",
     "Severe renal impairment mandates opioid dose reduction."),
    # 21
    ("SSRIs are used cautiously in geriatric patients due to hyponatremia risk.",
     "SSRIs are used cautiously in adolescent patients due to hyponatremia risk.",
     "Geriatric patients on SSRIs need monitoring for hyponatremia."),
    # 22
    ("Antipsychotics are used cautiously in geriatric patients with dementia due to mortality risk.",
     "Antipsychotics are used cautiously in adolescent patients with dementia due to mortality risk.",
     "Geriatric dementia patients need caution with antipsychotic therapy."),
    # 23
    ("Chemotherapy dose reduction occurs in patients with severe neutropenia.",
     "Chemotherapy dose reduction occurs in patients with severe leukocytosis.",
     "Severe neutropenia mandates chemotherapy dose adjustment."),
    # 24
    ("Rituximab requires hepatitis B screening in patients with prior HBV exposure.",
     "Rituximab requires hepatitis B screening in patients with prior HCV exposure.",
     "Prior HBV exposure mandates screening before rituximab therapy."),
    # 25
    ("Isoniazid dosing considers acetylator status which affects toxicity risk.",
     "Isoniazid dosing considers hydroxylator status which affects toxicity risk.",
     "Acetylator phenotype affects isoniazid toxicity risk."),
    # 26
    ("Warfarin monitoring intensifies in patients with recent high INR readings.",
     "Warfarin monitoring intensifies in patients with recent low PT readings.",
     "Recent high INR readings mandate closer warfarin monitoring."),
    # 27
    ("Insulin therapy in gestational diabetes uses regimens safe during pregnancy.",
     "Insulin therapy in gestational cirrhosis uses regimens safe during pregnancy.",
     "Gestational diabetes calls for pregnancy-safe insulin regimens."),
    # 28
    ("Beta-2 agonists caution required in patients with severe hypokalemia.",
     "Beta-2 agonists caution required in patients with severe hypernatremia.",
     "Severe hypokalemia mandates caution with beta-2 agonist therapy."),
    # 29
    ("Loop diuretic caution required in patients with severe hyponatremia.",
     "Loop diuretic caution required in patients with severe hyperglycemia.",
     "Severe hyponatremia mandates caution with loop diuretic therapy."),
    # 30
    ("ACE inhibitor caution required in patients with baseline hyperkalemia.",
     "ACE inhibitor caution required in patients with baseline hypercalcemia.",
     "Baseline hyperkalemia mandates caution when starting ACE inhibitors."),
    # 31
    ("Metronidazole use is avoided in patients with recent alcohol consumption due to disulfiram-like reaction.",
     "Metronidazole use is avoided in patients with recent caffeine consumption due to disulfiram-like reaction.",
     "Recent alcohol intake rules out metronidazole because of disulfiram-like reaction."),
    # 32
    ("Warfarin therapy monitoring intensifies with concurrent St John's wort use.",
     "Warfarin therapy monitoring intensifies with concurrent chamomile tea use.",
     "Concurrent St John's wort use mandates closer warfarin monitoring."),
    # 33
    ("SSRIs are used cautiously with concurrent tramadol due to serotonin syndrome risk.",
     "SSRIs are used cautiously with concurrent acetaminophen due to serotonin syndrome risk.",
     "Concurrent tramadol use with SSRIs raises serotonin syndrome risk."),
    # 34
    ("Vancomycin infusion rate is slowed in patients with prior red-man syndrome.",
     "Vancomycin infusion rate is slowed in patients with prior serotonin syndrome.",
     "Prior red-man syndrome mandates slower vancomycin infusion."),
    # 35
    ("Penicillins are avoided in patients with prior severe anaphylactic reaction.",
     "Penicillins are avoided in patients with prior severe allergic rhinitis reaction.",
     "Prior anaphylactic reaction to penicillin rules out re-exposure."),
    # 36
    ("Sulfa antibiotics are avoided in patients with prior Stevens-Johnson syndrome.",
     "Sulfa antibiotics are avoided in patients with prior mild urticaria syndrome.",
     "Prior Stevens-Johnson syndrome rules out sulfa antibiotic use."),
    # 37
    ("Iodinated contrast is avoided in patients with prior severe contrast reactions.",
     "Iodinated contrast is avoided in patients with prior mild contrast reactions.",
     "Prior severe contrast reactions rule out iodinated contrast administration."),
    # 38
    ("NSAIDs are avoided in patients with severe heart failure due to fluid retention.",
     "NSAIDs are avoided in patients with severe liver hemangioma due to fluid retention.",
     "Severe heart failure rules out NSAID use because of fluid retention."),
    # 39
    ("Metformin is held in patients with active metabolic acidosis and lactic acid elevation.",
     "Metformin is held in patients with active respiratory alkalosis and lactic acid elevation.",
     "Active metabolic acidosis with lactic acid elevation mandates holding metformin therapy."),
    # 40
    ("Corticosteroid dosing increases stress-dose coverage in diabetic patients undergoing surgery.",
     "Corticosteroid dosing increases stress-dose coverage in nondiabetic patients undergoing surgery.",
     "Chronic steroid users need stress-dose coverage for surgery."),
    # 41
    ("Basal insulin adjustment intensifies in patients with hypoglycemia unawareness.",
     "Basal insulin adjustment intensifies in patients with hyperglycemia unawareness.",
     "Hypoglycemia unawareness mandates careful basal insulin adjustment."),
    # 42
    ("Anticoagulation therapy is held in patients with active peptic ulcer bleeding.",
     "Anticoagulation therapy is held in patients with active hemorrhoid inflammation.",
     "Active peptic ulcer bleeding mandates holding anticoagulation."),
    # 43
    ("Antiplatelets are held in patients with recent gastrointestinal hemorrhage.",
     "Antiplatelets are held in patients with recent gastrointestinal reflux.",
     "Recent GI hemorrhage mandates holding antiplatelet therapy."),
    # 44
    ("Colchicine dose is reduced in patients with severe renal impairment on strong CYP3A4 inhibitors.",
     "Colchicine dose is reduced in patients with severe cardiac impairment on strong CYP3A4 inhibitors.",
     "Severe renal impairment plus CYP3A4 inhibitors mandate colchicine dose reduction."),
    # 45
    ("Digoxin toxicity monitoring intensifies with concurrent amiodarone therapy.",
     "Digoxin toxicity monitoring intensifies with concurrent atorvastatin therapy.",
     "Concurrent amiodarone use with digoxin mandates closer toxicity monitoring."),
    # 46
    ("Chemotherapy timing considers absolute neutrophil count in patients with prior neutropenic events.",
     "Chemotherapy timing considers absolute platelet count in patients with prior neutropenic events.",
     "Prior neutropenic events mandate absolute neutrophil count monitoring."),
    # 47
    ("Immunotherapy considers thyroid function in patients with baseline thyroid disease.",
     "Immunotherapy considers renal function in patients with baseline thyroid disease.",
     "Baseline thyroid disease mandates thyroid function monitoring on immunotherapy."),
    # 48
    ("Metformin therapy screens for vitamin B12 deficiency in long-term users.",
     "Metformin therapy screens for vitamin D deficiency in long-term users.",
     "Long-term metformin use mandates vitamin B12 deficiency screening."),
    # 49
    ("Long-term proton pump inhibitor therapy monitors for magnesium deficiency.",
     "Long-term proton pump inhibitor therapy monitors for calcium excess.",
     "Chronic PPI use mandates magnesium deficiency monitoring."),
    # 50
    ("Amiodarone therapy screens for pulmonary toxicity in long-term users.",
     "Amiodarone therapy screens for auditory toxicity in long-term users.",
     "Long-term amiodarone use mandates pulmonary toxicity screening."),
]


# ============================================================
# DRUG_SUBSTITUTION — 50 triples (LASA name-confusion or class-crossing)
# ============================================================

DRUG_SUBSTITUTION_TRIPLES = [
    # 1
    ("Hydroxyzine is used for adult acute anxiety and pruritus.",
     "Hydralazine is used for adult acute anxiety and pruritus.",
     "Adult acute anxiety with pruritus is treated with hydroxyzine."),
    # 2
    ("Prednisone is a systemic corticosteroid used for autoimmune inflammation.",
     "Primidone is a systemic corticosteroid used for autoimmune inflammation.",
     "Autoimmune inflammation is treated with the systemic corticosteroid prednisone."),
    # 3
    ("Cephalexin is a first-generation cephalosporin used for skin infections.",
     "Cyclosporine is a first-generation cephalosporin used for skin infections.",
     "Skin infections respond to the first-generation cephalosporin cephalexin."),
    # 4
    ("Chlorpromazine is a typical antipsychotic used for acute psychosis.",
     "Chlorpropamide is a typical antipsychotic used for acute psychosis.",
     "Acute psychosis is treated with the typical antipsychotic chlorpromazine."),
    # 5
    ("Bupropion is an atypical antidepressant used for major depressive disorder.",
     "Buspirone is an atypical antidepressant used for major depressive disorder.",
     "Major depressive disorder is treated with the atypical antidepressant bupropion."),
    # 6
    ("Tramadol is a weak opioid used for moderate pain management.",
     "Trazodone is a weak opioid used for moderate pain management.",
     "Moderate pain is treated with the weak opioid tramadol."),
    # 7
    ("Vinblastine is a vinca alkaloid used for Hodgkin lymphoma chemotherapy.",
     "Venlafaxine is a vinca alkaloid used for Hodgkin lymphoma chemotherapy.",
     "Hodgkin lymphoma is treated with the vinca alkaloid vinblastine."),
    # 8
    ("Zolpidem is a non-benzodiazepine hypnotic used for short-term insomnia.",
     "Zoloft is a non-benzodiazepine hypnotic used for short-term insomnia.",
     "Short-term insomnia is treated with the non-benzodiazepine hypnotic zolpidem."),
    # 9
    ("Lorazepam is a short-acting benzodiazepine used for acute anxiety.",
     "Ipratropium is a short-acting benzodiazepine used for acute anxiety.",
     "Acute anxiety is treated with the benzodiazepine lorazepam."),
    # 10
    ("Alprazolam is a benzodiazepine used for panic disorder treatment.",
     "Ranitidine is a benzodiazepine used for panic disorder treatment.",
     "Panic disorder is treated with the benzodiazepine alprazolam."),
    # 11
    ("Clonidine is a central alpha-2 agonist used for treatment-resistant hypertension.",
     "Clonazepam is a central alpha-2 agonist used for treatment-resistant hypertension.",
     "Treatment-resistant hypertension responds to the alpha-2 agonist clonidine."),
    # 12
    ("Amphetamine is a stimulant used for adult attention-deficit hyperactivity disorder.",
     "Propranolol is a stimulant used for adult attention-deficit hyperactivity disorder.",
     "Adult ADHD is treated with the stimulant amphetamine."),
    # 13
    ("Celecoxib is a COX-2 selective NSAID used for osteoarthritis pain.",
     "Fosphenytoin is a COX-2 selective NSAID used for osteoarthritis pain.",
     "Osteoarthritis pain responds to the COX-2 selective NSAID celecoxib."),
    # 14
    ("Doxazosin is an alpha-1 blocker used for adult benign prostatic hyperplasia.",
     "Warfarin is an alpha-1 blocker used for adult benign prostatic hyperplasia.",
     "Adult BPH is treated with the alpha-1 blocker doxazosin."),
    # 15
    ("Amlodipine is a calcium channel blocker used for essential hypertension.",
     "Thiothixene is a calcium channel blocker used for essential hypertension.",
     "Essential hypertension is treated with the calcium channel blocker amlodipine."),
    # 16
    ("Cetirizine is a second-generation antihistamine used for allergic rhinitis.",
     "Olanzapine is a second-generation antihistamine used for allergic rhinitis.",
     "Allergic rhinitis is treated with the second-generation antihistamine cetirizine."),
    # 17
    ("Fluoxetine is a selective serotonin reuptake inhibitor used for major depressive disorder.",
     "Fluoxymesterone is a selective serotonin reuptake inhibitor used for major depressive disorder.",
     "MDD is treated with the SSRI fluoxetine."),
    # 18
    ("Methylphenidate is a stimulant used for adult narcolepsy.",
     "Rifampin is a stimulant used for adult narcolepsy.",
     "Adult narcolepsy is treated with the stimulant methylphenidate."),
    # 19
    ("Ibuprofen is an NSAID used for adult mild to moderate pain.",
     "Bupropion is an NSAID used for adult mild to moderate pain.",
     "Adult mild pain responds to the NSAID ibuprofen."),
    # 20
    ("Regular insulin is a short-acting insulin used for adult postprandial glycemic control.",
     "Regular heparin is a short-acting insulin used for adult postprandial glycemic control.",
     "Adult postprandial glucose is controlled with short-acting regular insulin."),
    # 21
    ("Colchicine is used for adult acute gout flare treatment.",
     "Cyclobenzaprine is used for adult acute gout flare treatment.",
     "Adult acute gout flare responds to colchicine therapy."),
    # 22
    ("Losartan is an angiotensin receptor blocker used for essential hypertension.",
     "Omeprazole is an angiotensin receptor blocker used for essential hypertension.",
     "Essential hypertension responds to the ARB losartan."),
    # 23
    ("Donepezil is a cholinesterase inhibitor used for adult Alzheimer disease.",
     "Rabeprazole is a cholinesterase inhibitor used for adult Alzheimer disease.",
     "Adult Alzheimer disease is treated with the cholinesterase inhibitor donepezil."),
    # 24
    ("Oxycodone is a strong opioid used for adult severe acute pain.",
     "Oxycontin extended-release is an alternative formulation of the same drug family.",
     "Adult severe acute pain responds to the strong opioid oxycodone."),
    # 25
    ("Leucovorin is used as a folate rescue for adult methotrexate toxicity.",
     "Bupropion sustained-release is used as a folate rescue for adult methotrexate toxicity.",
     "Adult methotrexate toxicity is managed with folate rescue via leucovorin."),
    # 26
    ("Morphine is a strong opioid used for adult acute cancer pain.",
     "Metoprolol is a strong opioid used for adult acute cancer pain.",
     "Adult acute cancer pain responds to the strong opioid morphine."),
    # 27
    ("Metoprolol is a cardioselective beta-blocker used for adult heart failure.",
     "Misoprostol is a cardioselective beta-blocker used for adult heart failure.",
     "Adult heart failure is treated with the cardioselective beta-blocker metoprolol."),
    # 28
    ("Valsartan is an angiotensin receptor blocker used for adult heart failure.",
     "Fluconazole is an angiotensin receptor blocker used for adult heart failure.",
     "Adult heart failure responds to the ARB valsartan."),
    # 29
    ("Ceftriaxone is a third-generation cephalosporin used for adult bacterial meningitis.",
     "Calcitriol is a third-generation cephalosporin used for adult bacterial meningitis.",
     "Adult bacterial meningitis is treated with the third-generation cephalosporin ceftriaxone."),
    # 30
    ("Zidovudine is a nucleoside reverse-transcriptase inhibitor used for adult HIV.",
     "Ritonavir is a nucleoside reverse-transcriptase inhibitor used for adult HIV.",
     "Adult HIV is treated with the NRTI zidovudine."),
    # 31
    ("Fluconazole is a triazole antifungal used for adult candidemia.",
     "Difluprednate is a triazole antifungal used for adult candidemia.",
     "Adult candidemia is treated with the triazole antifungal fluconazole."),
    # 32
    ("Omeprazole is a proton pump inhibitor used for adult peptic ulcer disease.",
     "Fluoxetine is a proton pump inhibitor used for adult peptic ulcer disease.",
     "Adult peptic ulcer disease responds to the PPI omeprazole."),
    # 33
    ("Tofacitinib is a JAK inhibitor used for adult rheumatoid arthritis.",
     "Alprazolam is a JAK inhibitor used for adult rheumatoid arthritis.",
     "Adult rheumatoid arthritis is treated with the JAK inhibitor tofacitinib."),
    # 34
    ("Sertraline is a selective serotonin reuptake inhibitor used for panic disorder.",
     "Simvastatin is a selective serotonin reuptake inhibitor used for panic disorder.",
     "Panic disorder responds to the SSRI sertraline."),
    # 35
    ("Diphenoxylate is an opioid antidiarrheal used for adult chronic diarrhea.",
     "Lamotrigine is an opioid antidiarrheal used for adult chronic diarrhea.",
     "Adult chronic diarrhea is treated with the opioid antidiarrheal diphenoxylate."),
    # 36
    ("Mometasone is an intranasal corticosteroid used for adult allergic rhinitis.",
     "Ketoconazole is an intranasal corticosteroid used for adult allergic rhinitis.",
     "Adult allergic rhinitis is treated with the intranasal corticosteroid mometasone."),
    # 37
    ("Fluticasone-salmeterol is a combination inhaler used for adult persistent asthma.",
     "Simvastatin-niacin is a combination inhaler used for adult persistent asthma.",
     "Adult persistent asthma is treated with the fluticasone-salmeterol combination inhaler."),
    # 38
    ("Bicalutamide is an androgen receptor antagonist used for adult prostate cancer.",
     "Losartan is an androgen receptor antagonist used for adult prostate cancer.",
     "Adult prostate cancer is treated with the androgen receptor antagonist bicalutamide."),
    # 39
    ("Fosphenytoin is a prodrug antiepileptic used for adult status epilepticus.",
     "Citalopram is a prodrug antiepileptic used for adult status epilepticus.",
     "Adult status epilepticus responds to the prodrug antiepileptic fosphenytoin."),
    # 40
    ("Riluzole is a glutamate modulator used for adult amyotrophic lateral sclerosis.",
     "Lepirudin is a glutamate modulator used for adult amyotrophic lateral sclerosis.",
     "Adult ALS is treated with the glutamate modulator riluzole."),
    # 41
    ("Bisoprolol is a cardioselective beta-blocker used for adult heart failure.",
     "Zolpidem is a cardioselective beta-blocker used for adult heart failure.",
     "Adult heart failure responds to the cardioselective beta-blocker bisoprolol."),
    # 42
    ("Labetalol is a mixed alpha-beta blocker used for adult hypertensive emergency.",
     "Pentoxifylline is a mixed alpha-beta blocker used for adult hypertensive emergency.",
     "Adult hypertensive emergency is treated with the mixed alpha-beta blocker labetalol."),
    # 43
    ("Methylphenidate is a stimulant used for adult narcolepsy management.",
     "Methadone is a stimulant used for adult narcolepsy management.",
     "Adult narcolepsy responds to the stimulant methylphenidate."),
    # 44
    ("Formoterol is a long-acting beta-2 agonist used for adult asthma maintenance.",
     "Ketorolac is a long-acting beta-2 agonist used for adult asthma maintenance.",
     "Adult asthma maintenance uses the long-acting beta-2 agonist formoterol."),
    # 45
    ("Dolasetron is a serotonin receptor antagonist used for adult chemotherapy nausea.",
     "Donepezil is a serotonin receptor antagonist used for adult chemotherapy nausea.",
     "Adult chemo nausea responds to the serotonin receptor antagonist dolasetron."),
    # 46
    ("Atorvastatin is an HMG-CoA reductase inhibitor used for adult hyperlipidemia.",
     "Tamoxifen is an HMG-CoA reductase inhibitor used for adult hyperlipidemia.",
     "Adult hyperlipidemia is treated with the HMG-CoA reductase inhibitor atorvastatin."),
    # 47
    ("Pravastatin is a statin used for adult hyperlipidemia in liver-transplant patients.",
     "Prazosin is a statin used for adult hyperlipidemia in liver-transplant patients.",
     "Adult hyperlipidemia in liver-transplant patients responds to the statin pravastatin."),
    # 48
    ("Rosuvastatin is a high-potency statin used for adult hyperlipidemia.",
     "Rosiglitazone is a high-potency statin used for adult hyperlipidemia.",
     "Adult hyperlipidemia responds to the high-potency statin rosuvastatin."),
    # 49
    ("Aripiprazole is an atypical antipsychotic used for adult schizophrenia maintenance.",
     "Aripiprazole is a proton pump inhibitor used for adult peptic ulcer disease.",
     "Adult schizophrenia maintenance is treated with the atypical antipsychotic aripiprazole."),
    # 50
    ("Rifampin is a rifamycin antibiotic used for adult latent tuberculosis prophylaxis.",
     "Ritalin is a rifamycin antibiotic used for adult latent tuberculosis prophylaxis.",
     "Adult latent TB prophylaxis uses the rifamycin antibiotic rifampin."),
]


def build_cases(triples, trap_type, id_offset):
    cases = []
    for i, (a, b, c) in enumerate(triples):
        cases.append({
            "id": f"med_v2b_{trap_type}_{id_offset+i+1:03d}",
            "trap_type": trap_type,
            "clause_a": a,
            "clause_b": b,
            "clause_c": c,
        })
    return cases


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <output_json_path> [base_json_path]", file=sys.stderr)
        sys.exit(2)
    out_path = Path(sys.argv[1])

    # Load the 100 base triples to check no clause_a collisions
    existing_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "data" / "traps" / "medical_base_100.json"
    )
    existing = json.loads(existing_path.read_text())
    existing_a = {c["clause_a"].strip().lower() for c in existing["test_cases"]}

    new_cases = []
    new_cases += build_cases(DOSAGE_TRIPLES, "dosage_flip", id_offset=0)
    new_cases += build_cases(CONTRAINDICATION_TRIPLES, "contraindication_flip", id_offset=0)
    new_cases += build_cases(POPULATION_TRIPLES, "population_flip", id_offset=0)
    new_cases += build_cases(DRUG_SUBSTITUTION_TRIPLES, "drug_substitution", id_offset=0)

    print(f"[gen] built {len(new_cases)} new triples", file=sys.stderr)
    if len(new_cases) != 200:
        print(f"[gen] WARN: expected 200, got {len(new_cases)}", file=sys.stderr)

    print(f"[gen] checking distinctness against 100 existing...", file=sys.stderr)
    check_distinct(new_cases, existing_a)
    print(f"[gen] distinctness OK", file=sys.stderr)

    print(f"[gen] checking shape invariants (8-30 words, matched)...", file=sys.stderr)
    check_shape(new_cases)
    print(f"[gen] shape OK", file=sys.stderr)

    # Merge with existing 100 to form the 300-triple pack
    merged = existing["test_cases"] + new_cases
    output = dict(existing)  # keep metadata
    output["test_cases"] = merged
    output["_shape_contract"] = (
        "All A/B/C 8-30 words, matched-shape (max/min word count <= 1.7). "
        "B differs from A by one meaningful term reversing meaning. "
        "C paraphrases A in different surface wording, same intent."
    )
    output["_taxonomy_counts"] = {
        "dosage_flip": len([c for c in merged if c["trap_type"] == "dosage_flip"]),
        "contraindication_flip": len([c for c in merged if c["trap_type"] == "contraindication_flip"]),
        "population_flip": len([c for c in merged if c["trap_type"] == "population_flip"]),
        "drug_substitution": len([c for c in merged if c["trap_type"] == "drug_substitution"]),
    }
    print(f"[gen] taxonomy counts: {output['_taxonomy_counts']}", file=sys.stderr)
    print(f"[gen] total merged: {len(merged)}", file=sys.stderr)

    out_path.write_text(json.dumps(output, indent=2))
    print(f"[gen] wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
