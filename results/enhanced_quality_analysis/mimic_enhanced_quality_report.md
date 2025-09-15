# MIMIC 增强聚类和问题质量分析报告

生成时间: 2025-09-15 01:10:16.506382
使用训练好的CQG-MBQA模型计算真实logit分数

## 1. 聚类质量分析（含模型logit统计）

### Cluster 189 (共 3046 个文档)

#### 样本文档:

**POSITIVE (ID: 140969)**
Logit统计: 平均=-1.422, 最大=15.954, 最小=-15.237, 标准差=1.952
Name: Unit No: Date of Birth: Sex: M Allergies: Patient recorded as having No Known Allergies to Drugs Attending: Chief Complaint: Chest pain Major Surgical or Invasive Procedure: None History of Pres...

**POSITIVE (ID: 536743)**
Logit统计: 平均=-1.440, 最大=16.136, 最小=-15.368, 标准差=1.929
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: cp Major Surgical or Invasive Procedure: cath lab History of Present Illness: ...

**POSITIVE (ID: 1219045)**
Logit统计: 平均=-1.392, 最大=15.884, 最小=-14.955, 标准差=1.944
Name: Unit No: Date of Birth: Sex: M Allergies: No Allergies/ADRs on File Attending: Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: Cardiac Catheterization History of Present Illnes...

**POSITIVE (ID: 200362)**
Logit统计: 平均=-1.369, 最大=15.383, 最小=-14.609, 标准差=1.903
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Codeine / Nitroglycerin Attending: Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: None History of Present Illness: yo m...

**POSITIVE (ID: 206620)**
Logit统计: 平均=-1.319, 最大=16.063, 最小=-15.034, 标准差=1.927
Name: . Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Chest pain Major Surgical or Invasive Procedure: - Percutaneous Coronary Int...

**POSITIVE (ID: 1625330)**
Logit统计: 平均=-1.383, 最大=15.873, 最小=-15.018, 标准差=1.960
Name: Unit No: Date of Birth: Sex: M Allergies: Cozaar / Ace Inhibitors Attending: Complaint: Chest pain Major Surgical or Invasive Procedure: None. History of Present Illness: M with h/o CABG s/p ICd...

**EASY_NEGATIVE (ID: 1041357)**
Logit统计: 平均=-1.326, 最大=16.084, 最小=-15.083, 标准差=1.956
Name: Unit No: Date of Birth: Sex: M Allergies: lisinopril Attending: . Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: -Cardiac catheterization with Drug eluting stents x2 to Left c...

**EASY_NEGATIVE (ID: 172099)**
Logit统计: 平均=-1.400, 最大=16.342, 最小=-15.354, 标准差=1.974
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: None History of Present Illne...

**EASY_NEGATIVE (ID: 1148878)**
Logit统计: 平均=-1.419, 最大=16.025, 最小=-15.226, 标准差=1.943
Name: . Unit No: Date of Birth: Sex: M Allergies: Patient recorded as having No Known Allergies to Drugs Attending: Chief Complaint: chest pain Major Surgical or Invasive Procedure: Cardiac Catheteriz...

**EASY_NEGATIVE (ID: 295095)**
Logit统计: 平均=-1.414, 最大=15.955, 最小=-15.165, 标准差=1.931
Name: Unit No: Date of Birth: Sex: M Allergies: Nitroglycerin Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: none. History of Present Illness: This is a man with histor...

**EASY_NEGATIVE (ID: 295215)**
Logit统计: 平均=-1.389, 最大=16.031, 最小=-15.236, 标准差=1.943
Name: Unit No: Date of Birth: Sex: M Allergies: Nitroglycerin Attending: . Chief Complaint: Elevated Creatinine and atypical chest pain Major Surgical or Invasive Procedure: None History of Present Il...

**EASY_NEGATIVE (ID: 1457662)**
Logit统计: 平均=-1.373, 最大=15.765, 最小=-14.844, 标准差=1.931
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: none History of Present Illne...

**EASY_NEGATIVE (ID: 1027654)**
Logit统计: 平均=-1.353, 最大=15.994, 最小=-15.096, 标准差=1.973
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins Attending: . Chief Complaint: Chest Pressure/Intoxicated Major Surgical or Invasive Procedure: LHC History of Present Illness: male smoker w...

**EASY_NEGATIVE (ID: 1177332)**
Logit统计: 平均=-1.400, 最大=16.166, 最小=-15.409, 标准差=1.919
Name: Unit No: Date of Birth: Sex: M Allergies: Motrin / Codeine / Celebrex / IV Dye, Iodine Containing Contrast Media Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: Co...

**EASY_NEGATIVE (ID: 370315)**
Logit统计: 平均=-1.424, 最大=15.639, 最小=-14.979, 标准差=1.899
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: none History of Present Illne...

**EASY_NEGATIVE (ID: 588096)**
Logit统计: 平均=-1.412, 最大=16.165, 最小=-15.513, 标准差=1.977
Name: Unit No: Date of Birth: Sex: M Allergies: Cipro Attending: Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: Cardiac catheterization. No stents placed. History of Present Illness...

**EASY_NEGATIVE (ID: 1139736)**
Logit统计: 平均=-1.396, 最大=16.067, 最小=-15.174, 标准差=1.925
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: None History of Present Illness...

**EASY_NEGATIVE (ID: 1321166)**
Logit统计: 平均=-1.327, 最大=15.706, 最小=-14.828, 标准差=1.894
Name: Unit No: Date of Birth: Sex: M Allergies: Indomethacin Attending: . Chief Complaint: left sided chest pain Major Surgical or Invasive Procedure: cardiac catheterization History of Present Illnes...

**EASY_NEGATIVE (ID: 934796)**
Logit统计: 平均=-1.365, 最大=15.875, 最小=-15.183, 标准差=1.895
Name: Unit No: Date of Birth: Sex: M Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: Cardiac catheteriz...

**EASY_NEGATIVE (ID: 472988)**
Logit统计: 平均=-1.332, 最大=15.890, 最小=-14.962, 标准差=1.925
Name: Unit No: Date of Birth: Sex: M Allergies: Ace Inhibitors Attending: Chief Complaint: Chest pain Major Surgical or Invasive Procedure: None History of Present Illness: is a gentleman with history...

**EASY_NEGATIVE (ID: 1159805)**
Logit统计: 平均=-1.391, 最大=15.920, 最小=-15.171, 标准差=1.952
Name: Unit No: Date of Birth: Sex: M Allergies: Latex Attending: . Chief Complaint: Dyspnea and chest pain Major Surgical or Invasive Procedure: None History of Present Illness: y/o M cirrhosis second...

**EASY_NEGATIVE (ID: 1618580)**
Logit统计: 平均=-1.413, 最大=16.099, 最小=-15.369, 标准差=1.915
Name: Unit No: Date of Birth: Sex: M Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: Nuclear stress tes...

**EASY_NEGATIVE (ID: 1524540)**
Logit统计: 平均=-1.393, 最大=15.814, 最小=-14.947, 标准差=1.900
Name: Unit No: Date of Birth: Sex: M Allergies: Tylenol / gabapentin Attending: . Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: NUCLEAR CARDIAC PERFUSION TEST RESULTS: Mild fixed s...

**EASY_NEGATIVE (ID: 446521)**
Logit统计: 平均=-1.372, 最大=16.185, 最小=-15.384, 标准差=2.002
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Codeine / Iodine / Dicloxacillin Attending: . Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: None History of Present Il...

**HARD_NEGATIVE (ID: 1587666)**
Logit统计: 平均=-1.421, 最大=15.728, 最小=-14.981, 标准差=1.914
Name: Unit No: Date of Birth: Sex: M Allergies: Erythromycin Base Attending: Chief Complaint: chest discomfort Major Surgical or Invasive Procedure: none History of Present Illness: Patient is a old m...

**HARD_NEGATIVE (ID: 674983)**
Logit统计: 平均=-1.407, 最大=16.068, 最小=-15.302, 标准差=1.936
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Optiray 350 / Clindamycin Attending: Chief Complaint: Chest pain Major Surgical or Invasive Procedure: Cardiac catheterization History of ...

**HARD_NEGATIVE (ID: 860710)**
Logit统计: 平均=-1.395, 最大=15.537, 最小=-14.715, 标准差=1.906
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Chest Pain Decreased exercise tolerance Major Surgical or Invasive Procedure: ...

**HARD_NEGATIVE (ID: 1483620)**
Logit统计: 平均=-1.382, 最大=15.825, 最小=-15.014, 标准差=1.928
Name: Unit No: Date of Birth: Sex: M Allergies: Diovan / spironolactone Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: NONE History of Present Illness: Mr. is a y/o man...

**HARD_NEGATIVE (ID: 1434176)**
Logit统计: 平均=-1.420, 最大=16.128, 最小=-15.273, 标准差=1.944
Name: Unit No: Date of Birth: Sex: M Allergies: Patient recorded as having No Known Allergies to Drugs Attending: Chief Complaint: Chest Pain Major Surgical or Invasive Procedure: Cardiac Catheterizat...

**HARD_NEGATIVE (ID: 805209)**
Logit统计: 平均=-1.353, 最大=16.029, 最小=-15.293, 标准差=1.957
Name: Unit No: Date of Birth: Sex: M Allergies: Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: Coronary angiography History of Present Illness: h/o CAD s/p CABG (), ETO...

**HARD_NEGATIVE (ID: 963013)**
Logit统计: 平均=-1.375, 最大=15.670, 最小=-14.748, 标准差=1.919
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Chest pain with exertion Major Surgical or Invasive Procedure: None History of...

**HARD_NEGATIVE (ID: 1120744)**
Logit统计: 平均=-1.439, 最大=15.924, 最小=-15.153, 标准差=1.944
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Codeine / Sudafed / prochlorperazine Attending: . Chief Complaint: Chest pain. Major Surgical or Invasive Procedure: Cardiac catheterizati...

**HARD_NEGATIVE (ID: 990088)**
Logit统计: 平均=-1.419, 最大=16.148, 最小=-15.341, 标准差=1.938
Name: Unit No: Date of Birth: Sex: M : MEDICINE Allergies: metformin Attending: . Chief Complaint: atypical chest pain Major Surgical or Invasive Procedure: atypical chest pain - no intervention, incr...

**HARD_NEGATIVE (ID: 283142)**
Logit统计: 平均=-1.376, 最大=15.844, 最小=-14.946, 标准差=1.933
Name: Unit No: Date of Birth: Sex: M Allergies: Allergy / Egg / topiramate Attending: Chief Complaint: chest pain Major Surgical or Invasive Procedure: cardiac catheterization - History of Present Ill...

**HARD_NEGATIVE (ID: 552756)**
Logit统计: 平均=-1.343, 最大=16.042, 最小=-15.078, 标准差=1.966
Name: Unit No: Date of Birth: Sex: M Allergies: Iodine / shellfish derived Attending: . Chief Complaint: Chest pain Major Surgical or Invasive Procedure: none History of Present Illness: HISTORY OF PR...

**HARD_NEGATIVE (ID: 501623)**
Logit统计: 平均=-1.385, 最大=16.204, 最小=-15.480, 标准差=1.921
Name: Unit No: Date of Birth: Sex: M Allergies: Cipro Attending: . Chief Complaint: Gagging and left shoulder pain Major Surgical or Invasive Procedure: None History of Present Illness: y/o male with ...

**HARD_NEGATIVE (ID: 157299)**
Logit统计: 平均=-1.386, 最大=15.660, 最小=-14.838, 标准差=1.852
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins Attending: Chief Complaint: chest pressure, left arm tingling/numbness Major Surgical or Invasive Procedure: cardiac cath, diagnostic, showe...

**HARD_NEGATIVE (ID: 727165)**
Logit统计: 平均=-1.390, 最大=15.920, 最小=-15.195, 标准差=1.960
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Sulfa(Sulfonamide Antibiotics) / lisinopril / simvastatin Attending: Chief Complaint: Chest pain radiating to left arm Major Surgical or I...

**HARD_NEGATIVE (ID: 54976)**
Logit统计: 平均=-1.378, 最大=16.080, 最小=-15.128, 标准差=1.971
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: none History of Present Illne...

**HARD_NEGATIVE (ID: 1275631)**
Logit统计: 平均=-1.375, 最大=16.044, 最小=-15.065, 标准差=1.977
Name: Unit No: Date of Birth: Sex: M Allergies: Atorvastatin Attending: . Chief Complaint: chest pain Major Surgical or Invasive Procedure: Cardiac catheterization Placement of 3 drug eluting stents H...

**HARD_NEGATIVE (ID: 1191720)**
Logit统计: 平均=-1.309, 最大=15.281, 最小=-14.487, 标准差=1.866
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: Chief Complaint: Chest pain Major Surgical or Invasive Procedure: Left heart catheterization () w...

**HARD_NEGATIVE (ID: 516649)**
Logit统计: 平均=-1.384, 最大=16.000, 最小=-15.158, 标准差=1.971
Name: Unit No: Date of Birth: Sex: M Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Chest pain Major Surgical or Invasive Procedure: Cardiac catheterization Histo...

#### 相关问题:

1. Is there a reference to multiple vessel disease in the cardiac assessment? (来源文档: 14734)
2. Did the article indicate a need for urgent cardiac surgical revascularization? (来源文档: 14734)
3. Does the medication list contain a variety of prescribed medications tailored to individual needs? (来源文档: 14734)
4. Is there any indication of treatment involving heparin or nitroglycerin? (来源文档: 14734)
5. Is there documentation of a neurologic examination performed on the patient? (来源文档: 14734)
6. Are headaches described as mild to moderate? (来源文档: 14734)
7. Does the report mention that interventions led to symptom relief? (来源文档: 14734)
8. Was the patient's initial presentation consistent with a serious cardiac issue? (来源文档: 14734)
9. Does the article instruct to secure the Foley catheter at all times? (来源文档: 14734)
10. Is there a history of cardiac procedures, such as catheterization or surgery? (来源文档: 14734)

### Cluster 1194 (共 2452 个文档)

#### 样本文档:

**POSITIVE (ID: 1610708)**
Logit统计: 平均=-1.446, 最大=15.496, 最小=-15.276, 标准差=1.933
Name: Unit No: Date of Birth: Sex: F Allergies: Aspirin / Sulfa (Sulfonamide Antibiotics) Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: none History of Present Ill...

**POSITIVE (ID: 593788)**
Logit统计: 平均=-1.345, 最大=15.858, 最小=-15.231, 标准差=1.955
Name: Unit No: Date of Birth: Sex: F Allergies: Penicillins / latex / morphine / aspirin / eggs / shellfish derived / citoxin / azathorapine / Keflex / strawberries Attending: . Chief Complaint: Abdom...

**POSITIVE (ID: 24007)**
Logit统计: 平均=-1.444, 最大=16.071, 最小=-15.490, 标准差=1.954
Name: Unit No: Date of Birth: Sex: F Allergies: Penicillins Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: None History of Present Illness: Patient is a female with...

**POSITIVE (ID: 196591)**
Logit统计: 平均=-1.499, 最大=15.960, 最小=-15.563, 标准差=1.918
Name: Unit No: Date of Birth: Sex: F Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: four day history of abdominal pain Major Surgical or Invasive Proce...

**POSITIVE (ID: 168093)**
Logit统计: 平均=-1.421, 最大=15.652, 最小=-15.590, 标准差=1.951
Name: Unit No: Date of Birth: Sex: F Allergies: Penicillins / Sulfa (Sulfonamide Antibiotics) / gluten free Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: : Open il...

**POSITIVE (ID: 621683)**
Logit统计: 平均=-1.464, 最大=15.821, 最小=-15.383, 标准差=1.946
Name: Unit No: Date of Birth: Sex: F Allergies: Penicillins / morphine Attending: . Chief Complaint: abd pain Major Surgical or Invasive Procedure: none History of Present Illness: with a history of c...

**EASY_NEGATIVE (ID: 191781)**
Logit统计: 平均=-1.436, 最大=15.697, 最小=-15.387, 标准差=1.933
Name: Unit No: Date of Birth: Sex: F Allergies: adhesive tape / NSAIDS Attending: . Chief Complaint: Diarrhea Major Surgical or Invasive Procedure: None this hospitalization History of Present Illness...

**EASY_NEGATIVE (ID: 91140)**
Logit统计: 平均=-1.385, 最大=15.853, 最小=-15.336, 标准差=1.961
Name: Unit No: Date of Birth: Sex: F Allergies: Nsaids / Motrin / Compazine / Voltaren Attending: . Chief Complaint: Abdominal pain, diarrhea. Major Surgical or Invasive Procedure: none History of Pre...

**EASY_NEGATIVE (ID: 914829)**
Logit统计: 平均=-1.372, 最大=15.858, 最小=-15.218, 标准差=1.966
Name: Unit No: Date of Birth: Sex: F Allergies: Lyrica / lisinopril / Bactrim Attending: . Chief Complaint: abdominal pain Major Surgical or Invasive Procedure: None History of Present Illness: Mrs. i...

**EASY_NEGATIVE (ID: 1421826)**
Logit统计: 平均=-1.480, 最大=15.770, 最小=-15.513, 标准差=1.939
Name: Unit No: Date of Birth: Sex: F Allergies: No Known Allergies / Adverse Drug Reactions Attending: Chief Complaint: abdominal pain Major Surgical or Invasive Procedure: None History of Present Ill...

**EASY_NEGATIVE (ID: 640101)**
Logit统计: 平均=-1.449, 最大=15.815, 最小=-15.442, 标准差=1.943
Name: . Unit No: Date of Birth: Sex: F Allergies: Cefaclor / Compazine / Reglan / OxyContin / Percocet Attending: Chief Complaint: Abdominal pain, nausea, vomiting, diarrhea Major Surgical or Invasive...

**EASY_NEGATIVE (ID: 752030)**
Logit统计: 平均=-1.475, 最大=15.650, 最小=-15.457, 标准差=1.934
Name: Unit No: Date of Birth: Sex: F Allergies: Tolmetin Attending: . Chief Complaint: abdominal pain, back pain Major Surgical or Invasive Procedure: none History of Present Illness: w/RA on predniso...

**EASY_NEGATIVE (ID: 1340793)**
Logit统计: 平均=-1.383, 最大=15.464, 最小=-15.103, 标准差=1.909
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Nexium Packet / Prilosec OTC / Augmentin / Macrobid / Prevacid Attending: . Chief Complaint: abdominal pain Major Surg...

**EASY_NEGATIVE (ID: 576054)**
Logit统计: 平均=-1.369, 最大=15.439, 最小=-14.839, 标准差=1.918
Name: Unit No: Date of Birth: Sex: F Allergies: Motrin / Trazamine / Latex / Trazodone Attending: . Chief Complaint: right upper abdominal pain Major Surgical or Invasive Procedure: EGD History of Pre...

**EASY_NEGATIVE (ID: 1488999)**
Logit统计: 平均=-1.394, 最大=16.014, 最小=-15.376, 标准差=1.960
Name: Unit No: Date of Birth: Sex: F Allergies: Allopurinol And Derivatives / PhosLo Attending: . Chief Complaint: Nausea, abdominal pain Major Surgical or Invasive Procedure: None History of Present ...

**EASY_NEGATIVE (ID: 334242)**
Logit统计: 平均=-1.365, 最大=15.549, 最小=-15.338, 标准差=1.943
Name: Unit No: Date of Birth: Sex: F Allergies: Bactrim / Imitrex Attending: Chief Complaint: Abdominal pain, diarrhea Major Surgical or Invasive Procedure: None History of Present Illness: y/o F s/p ...

**EASY_NEGATIVE (ID: 1566405)**
Logit统计: 平均=-1.397, 最大=15.556, 最小=-15.164, 标准差=1.892
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Penicillins Attending: . Chief Complaint: n/v/abd pain Major Surgical or Invasive Procedure: Exploratory laparotomy an...

**EASY_NEGATIVE (ID: 1579576)**
Logit统计: 平均=-1.434, 最大=15.417, 最小=-15.069, 标准差=1.926
Name: Unit No: Date of Birth: Sex: F Allergies: ciprofloxacin / Codeine / Latex, Natural Rubber / naproxen / Tylenol Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: ...

**EASY_NEGATIVE (ID: 1301776)**
Logit统计: 平均=-1.396, 最大=15.752, 最小=-15.464, 标准差=1.944
Name: . Unit No: Date of Birth: Sex: F Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: abdominal pain Major Surgical or Invasive Procedure: none History...

**EASY_NEGATIVE (ID: 662214)**
Logit统计: 平均=-1.433, 最大=15.667, 最小=-15.526, 标准差=1.933
Name: Unit No: Date of Birth: Sex: F Allergies: Wellbutrin / Tramadol Attending: . Chief Complaint: abdominal pain diarrhea Major Surgical or Invasive Procedure: none History of Present Illness: The p...

**EASY_NEGATIVE (ID: 1302102)**
Logit统计: 平均=-1.395, 最大=15.332, 最小=-15.101, 标准差=1.903
Name: Unit No: Date of Birth: Sex: F Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Abdominal pain and diarrhea Major Surgical or Invasive Procedure: None History...

**EASY_NEGATIVE (ID: 1149507)**
Logit统计: 平均=-1.444, 最大=15.405, 最小=-15.172, 标准差=1.920
Name: Unit No: Date of Birth: Sex: M Allergies: Penicillins / Reglan / Codeine / aspirin / clindamycin Attending: . Chief Complaint: Dehydration/Abdominal Pain/Nausea/Vomiting Major Surgical or Invasi...

**EASY_NEGATIVE (ID: 501931)**
Logit统计: 平均=-1.408, 最大=16.061, 最小=-15.524, 标准差=1.931
Name: Unit No: Date of Birth: Sex: F Allergies: Demerol / Phenergan / Compazine / Effexor / Ms Attending: . Chief Complaint: Abdominal Pain Major Surgical or Invasive Procedure: Colonoscopy Angiogram ...

**EASY_NEGATIVE (ID: 246415)**
Logit统计: 平均=-1.451, 最大=15.438, 最小=-15.193, 标准差=1.894
Name: Unit No: Date of Birth: Sex: F Allergies: Dilaudid Attending: Complaint: Abdominal pain and nausea/vomiting Major Surgical or Invasive Procedure: None History of Present Illness: hx of refractor...

**HARD_NEGATIVE (ID: 252080)**
Logit统计: 平均=-1.434, 最大=15.351, 最小=-15.045, 标准差=1.917
Name: Unit No: Date of Birth: Sex: F Allergies: Neomycin / Polymyxin B / chicken Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: none History of Present Illness: Thi...

**HARD_NEGATIVE (ID: 1203195)**
Logit统计: 平均=-1.389, 最大=15.506, 最小=-15.325, 标准差=1.981
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Aspirin / Motrin / Bactrim / Compazine / Cipro / Ritalin / NSAIDS (Non-Steroidal Anti-Inflammatory Drug) / Zofran (as ...

**HARD_NEGATIVE (ID: 982495)**
Logit统计: 平均=-1.418, 最大=15.538, 最小=-15.190, 标准差=1.902
Name: Unit No: Date of Birth: Sex: F Allergies: Percocet / Morphine Attending: Chief Complaint: abdominal pain Major Surgical or Invasive Procedure: none History of Present Illness: YO F s/p multiple ...

**HARD_NEGATIVE (ID: 1183324)**
Logit统计: 平均=-1.370, 最大=15.538, 最小=-15.375, 标准差=1.979
Name: Unit No: Date of Birth: Sex: F Allergies: Zosyn Attending: . Chief Complaint: Fevers, vomiting, diarrhea Major Surgical or Invasive Procedure: None History of Present Illness: F w/ biliary cirrh...

**HARD_NEGATIVE (ID: 1156657)**
Logit统计: 平均=-1.413, 最大=15.310, 最小=-14.867, 标准差=1.897
Name: Unit No: Date of Birth: Sex: F Allergies: morphine / Penicillins / Sulfa (Sulfonamide Antibiotics) Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: None History...

**HARD_NEGATIVE (ID: 1292070)**
Logit统计: 平均=-1.433, 最大=15.775, 最小=-15.321, 标准差=1.924
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Penicillins / Topamax / adhesive / lisinopril Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive ...

**HARD_NEGATIVE (ID: 154782)**
Logit统计: 平均=-1.404, 最大=15.784, 最小=-15.656, 标准差=1.969
Name: Unit No: Date of Birth: Sex: F Allergies: naproxen Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: : Sigmoidoscopy : EGD History of Present Illness: Mrs. is a ...

**HARD_NEGATIVE (ID: 255273)**
Logit统计: 平均=-1.406, 最大=15.719, 最小=-15.411, 标准差=1.906
Name: Unit No: Date of Birth: Sex: F Allergies: Ciprofloxacin Attending: . Chief Complaint: Celiac artery pseudoaneurysm Major Surgical or Invasive Procedure: Resection of the celiac artery pseudoaneu...

**HARD_NEGATIVE (ID: 174524)**
Logit统计: 平均=-1.359, 最大=15.215, 最小=-14.697, 标准差=1.937
Name: Unit No: Date of Birth: Sex: F Allergies: Phenobarbital / Dilantin / Lamictal / Carafate / Amitriptyline Attending: . Chief Complaint: Abdominal pain, nausea Major Surgical or Invasive Procedure...

**HARD_NEGATIVE (ID: 1133915)**
Logit统计: 平均=-1.421, 最大=16.070, 最小=-15.470, 标准差=1.973
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfasalazine / Imitrex / trazodone / Augmentin Attending: . Chief Complaint: abdominal pain Major Surgical or Invasive Procedure: None History of Prese...

**HARD_NEGATIVE (ID: 936193)**
Logit统计: 平均=-1.419, 最大=15.937, 最小=-15.416, 标准差=1.933
Name: . Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Morphine Sulfate / Reglan Attending: Chief Complaint: Abdominal Pain Major Surgical or Invasive Procedure: None Hist...

**HARD_NEGATIVE (ID: 276632)**
Logit统计: 平均=-1.403, 最大=15.691, 最小=-15.394, 标准差=1.903
Name: Unit No: Date of Birth: Sex: F Allergies: Meperidine / Bactrim / Penicillins / Amoxicillin Attending: . Chief Complaint: abd pain since x 12hrs Major Surgical or Invasive Procedure: None History...

**HARD_NEGATIVE (ID: 680251)**
Logit统计: 平均=-1.468, 最大=15.806, 最小=-15.421, 标准差=1.953
Name: Unit No: Date of Birth: Sex: F Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: Abdominal Pain Major Surgical or Invasive Procedure: : Partial Gastrectomy, Bi...

**HARD_NEGATIVE (ID: 507909)**
Logit统计: 平均=-1.399, 最大=15.635, 最小=-15.340, 标准差=1.975
Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) Attending: . Chief Complaint: Abdominal Pain Major Surgical or Invasive Procedure: None History of Present Illness: HPI:...

**HARD_NEGATIVE (ID: 1483693)**
Logit统计: 平均=-1.429, 最大=15.662, 最小=-15.330, 标准差=1.933
Name: Unit No: Date of Birth: Sex: F Allergies: codeine / hydrocodone Attending: . Chief Complaint: Abdominal Pain Major Surgical or Invasive Procedure: None History of Present Illness: w/complex abdo...

**HARD_NEGATIVE (ID: 1247821)**
Logit统计: 平均=-1.452, 最大=15.855, 最小=-15.516, 标准差=1.949
Name: Unit No: Date of Birth: Sex: F Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: Abdominal pain Major Surgical or Invasive Procedure: None History o...

**HARD_NEGATIVE (ID: 367398)**
Logit统计: 平均=-1.363, 最大=15.912, 最小=-15.280, 标准差=1.940
Name: Unit No: Date of Birth: Sex: F Allergies: Keflex / Penicillins / Dicloxacillin / Morphine / Compazine / Reglan / Amicar / Verapamil / Ambien / Valtrex / Percocet / Vibramycin / doxycycline / Dem...

**HARD_NEGATIVE (ID: 1180422)**
Logit统计: 平均=-1.476, 最大=15.849, 最小=-15.751, 标准差=1.959
Name: Unit No: Date of Birth: Sex: F Allergies: Patient recorded as having No Known Allergies to Drugs Attending: . Chief Complaint: Abdominal Pain and Nausea Major Surgical or Invasive Procedure: Non...

#### 相关问题:

1. Is there mention of laboratory tests or imaging that contributed to patient management? (来源文档: 55235)
2. Does the article include a transfer to a higher level of care or specialty service? (来源文档: 55235)
3. Is there a mention of a disease or condition that requires further diagnostics in the article? (来源文档: 55235)
4. Does the article provide details about a patient's vital signs during their examination? (来源文档: 55235)
5. Is the treatment plan in the article related to surgical management of gallbladder issues? (来源文档: 55235)
6. Is there a mention of previous medical history related to liver conditions? (来源文档: 55235)
7. Does the patient have a significant acute abdominal condition? (来源文档: 55235)
8. Is there a history of surgical procedures indicated in the article? (来源文档: 55235)
9. Does the patient have a history of ulcerative colitis or Crohn's disease? (来源文档: 55235)
10. Does the article discuss a patient with symptoms related to inflammatory bowel disease? (来源文档: 55235)

### Cluster 825 (共 2356 个文档)

#### 样本文档:

**POSITIVE (ID: 536451)**
Logit统计: 平均=-1.444, 最大=14.108, 最小=-13.906, 标准差=2.066
1,000 unit Cap Oral 1 Capsule(s) Once Daily - *osteo 3 Once Daily - Imodium Relief 2 mg Tab qdaily - Oxycodone/acetaminophen tabs Q 4 hrs prn pain - Beclomethasone diproprionate 40mcg/actuation- 2 puf...

**POSITIVE (ID: 825015)**
Logit统计: 平均=-1.555, 最大=13.849, 最小=-13.739, 标准差=2.195
Tablet Sig: One (1) Tablet PO DAILY (Daily). Simvastatin 40 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Lisinopril 5 mg Tablet Sig: 5 Tablet PO DAILY (Daily). Clobetasol 05 % Cream Sig: One (1) Ap...

**POSITIVE (ID: 1341947)**
Logit统计: 平均=-1.499, 最大=14.172, 最小=-13.850, 标准差=2.254
(1) Tablet PO DAILY (Daily). amlodipine-benazepril mg Capsule Sig: One (1) Capsule PO once a day. calcium carbonate 200 mg calcium (500 mg) Tablet, Chewable Sig: One (1) Tablet, Chewable PO DAILY (Dai...

**POSITIVE (ID: 752537)**
Logit统计: 平均=-1.519, 最大=14.297, 最小=-14.173, 标准差=2.181
daily - omeprazole 20 mg Capsule, daily - senna 6 mg Tablet 1 daily - aspirin 325 mg Tablet daily - brimonidine 15 % Drops Sig: One (1) Drop Ophthalmic Q 12H - latanoprost 005 % Drops Sig: One (1) Dro...

**POSITIVE (ID: 210940)**
Logit统计: 平均=-1.587, 最大=14.408, 最小=-14.233, 标准差=2.199
Sig: Ten (10) units Subcutaneous at bedtime. Humalog Subcutaneous senna 6 mg Tablet Sig: One (1) Tablet PO BID (2 times a day) as needed for constipation. sevelamer carbonate 800 mg Tablet Sig: Two (2...

**POSITIVE (ID: 1282930)**
Logit统计: 平均=-1.560, 最大=14.192, 最小=-14.007, 标准差=2.226
Sig: One (1) Capsule PO DAILY (Daily). Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day). Senna 6 mg Tablet Sig: One (1) Tablet PO BID (2 times a day) as needed. Calcium Carbo...

**EASY_NEGATIVE (ID: 1566594)**
Logit统计: 平均=-1.544, 最大=14.479, 最小=-14.316, 标准差=2.154
mg Tablet Sig: One (1) Tablet PO BID (2 times a day). folic acid 1 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). levetiracetam 500 mg Tablet Sig: 5 Tablets PO BID (2 times a day). phenobarbital 100 ...

**EASY_NEGATIVE (ID: 277633)**
Logit统计: 平均=-1.513, 最大=14.113, 最小=-13.977, 标准差=2.167
(2) Tablet PO DAILY (Daily). Hexavitamin Tablet Sig: One (1) Cap PO DAILY (Daily). Atenolol 50 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Fluticasone 110 mcg/Actuation Aerosol Sig: Two (2) Puff I...

**EASY_NEGATIVE (ID: 152009)**
Logit统计: 平均=-1.453, 最大=14.165, 最小=-14.066, 标准差=2.131
sbp <110 or HR < emtricitabine-tenofovir 200-300 mg Tablet Sig: One (1) Tablet PO Q72H (every 72 hours): last . mycophenolate mofetil 500 mg Tablet Sig: One (1) Tablet PO BID (2 times a day). omeprazo...

**EASY_NEGATIVE (ID: 1528396)**
Logit统计: 平均=-1.550, 最大=14.626, 最小=-14.407, 标准差=2.206
PO DAILY (Daily). furosemide 40 mg Tablet Sig: One (1) Tablet PO DAILY (Daily) for 7 days: d/c when pedal edema resolved. metoprolol tartrate 25 mg Tablet Sig: 5 Tablet PO BID (2 times a day). simvast...

**EASY_NEGATIVE (ID: 53595)**
Logit统计: 平均=-1.555, 最大=14.304, 最小=-13.942, 标准差=2.232
HCl 4 % Drops Sig: One (1) Drop Ophthalmic Q8H (every 8 hours). docusate sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day). nitroglycerin 4 mg Tablet, Sublingual Sig: One (1) Sublingua...

**EASY_NEGATIVE (ID: 269167)**
Logit统计: 平均=-1.579, 最大=14.573, 最小=-14.502, 标准差=2.173
6 hours) as needed for shortness of breath. Theophylline 100 mg Tablet Sustained Release 12 hr Sig: One (1) Tablet Sustained Release 12 hr PO Q12H (every 12 hours) as needed for SOB/Asthma. Ezetimibe ...

**EASY_NEGATIVE (ID: 719455)**
Logit统计: 平均=-1.564, 最大=14.602, 最小=-14.289, 标准差=2.327
PO BID (2 times a day). B Complex-Vitamin C-Folic Acid 1 mg Capsule Sig: One (1) Cap PO DAILY (Daily). Atorvastatin 20 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Levothyroxine 175 mcg Tablet Sig:...

**EASY_NEGATIVE (ID: 485664)**
Logit统计: 平均=-1.539, 最大=14.304, 最小=-14.078, 标准差=2.311
Tablet PO DAILY (Daily). - Aspirin 325 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). - Calcium Carbonate 500 mg Tablet, Chewable Sig: One (1) Tablet, Chewable PO TID W/MEALS (3 TIMES A DAY WITH MEAL...

**EASY_NEGATIVE (ID: 1180070)**
Logit统计: 平均=-1.511, 最大=13.564, 最小=-13.387, 标准差=2.208
25 mg Tablet Sig: One (1) Tablet PO BID (2 times a day). Nitroglycerin 3 mg Tablet, Sublingual Sig: Tablet, Sublinguals Sublingual PRN (as needed) as needed for jaw pain. Omeprazole 20 mg Capsule, Del...

**EASY_NEGATIVE (ID: 1386904)**
Logit统计: 平均=-1.463, 最大=14.469, 最小=-14.215, 标准差=2.282
25 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Heparin IV continous as per sliding scale until goal PTT Discharge Medications: Albuterol Sulfate 5 mg /3 mL (083 %) Solution for Nebulization Sig: O...

**EASY_NEGATIVE (ID: 1381735)**
Logit统计: 平均=-1.553, 最大=14.428, 最小=-14.173, 标准差=2.334
IN CARE: MEDICATION CHANGES: none FOLLOW-UP APPOINTMENTS: with PCP on : amitriptyline 10 mg Tablet Sig: 5 Tablets PO HS (at bedtime). amlodipine 5 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). aripi...

**EASY_NEGATIVE (ID: 645601)**
Logit统计: 平均=-1.562, 最大=14.664, 最小=-14.481, 标准差=2.212
(at bedtime). Lisinopril 40 mg Tablet Sig: One (1) Tablet PO once a day. Norvasc 10 mg Tablet Sig: One (1) Tablet PO once a day. Metoprolol Succinate 25 mg Tablet Sustained Release 24 hr Sig: One (1) ...

**EASY_NEGATIVE (ID: 1165590)**
Logit统计: 平均=-1.530, 最大=14.322, 最小=-14.128, 标准差=2.256
(every 8 hours). Tamsulosin 4 mg Capsule, Sust. Release 24 hr Sig: One (1) Capsule, Sust. Release 24 hr PO HS (at bedtime). Warfarin 2 mg Tablet Sig: Three (3) Tablet PO once a day. Disp:*90 Tablet(s)...

**EASY_NEGATIVE (ID: 173980)**
Logit统计: 平均=-1.529, 最大=14.257, 最小=-13.927, 标准差=2.260
Tablet po warfarin 5 mg Tablet Sig: One (1) Tablet PO simethicone 80 mg Tablet 1 po qid prn gas cholecalciferol (vitamin D3) 800 unit Tablet 1po qday multivitamin 1 po qday furosemide 40 mg Tablet Sig...

**EASY_NEGATIVE (ID: 479976)**
Logit统计: 平均=-1.588, 最大=14.494, 最小=-14.145, 标准差=2.336
Sig: Two (2) Tablet PO TID (3 times a day) as needed for pain: Do not exceed 4 grams a day. Lisinopril 10 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Simvastatin 10 mg Tablet Sig: Two (2) Tablet P...

**EASY_NEGATIVE (ID: 1101910)**
Logit统计: 平均=-1.564, 最大=14.103, 最小=-13.918, 标准差=2.201
times a day). Cholecalciferol (Vitamin D3) 400 unit Tablet Sig: Two (2) Tablet PO DAILY (Daily). Aspirin, Buffered 325 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Artificial Tears Drops Sig: One (...

**EASY_NEGATIVE (ID: 5575)**
Logit统计: 平均=-1.532, 最大=14.384, 最小=-14.219, 标准差=2.235
6 mg Capsule Sig: One (1) Tablet PO DAILY (Daily). docusate sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day). dicloxacillin 250 mg Capsule Sig: One (1) Capsule PO Q6H (every 6 hours) ...

**EASY_NEGATIVE (ID: 1128863)**
Logit统计: 平均=-1.500, 最大=14.042, 最小=-13.806, 标准差=2.165
week. Atenolol 50 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day). Lamivudine 100 mg Tablet Sig: One (1) Tablet PO DAILY (Dai...

**HARD_NEGATIVE (ID: 800913)**
Logit统计: 平均=-1.481, 最大=14.073, 最小=-13.825, 标准差=2.225
Sig: One (1) Tablet PO DAILY (Daily). Lisinopril 5 mg Tablet Sig: 5 Tablet PO DAILY (Daily). Sotalol 80 mg Tablet Sig: One (1) Tablet PO BID (2 times a day). Mexiletine 150 mg Capsule Sig: One (1) Cap...

**HARD_NEGATIVE (ID: 1323225)**
Logit统计: 平均=-1.520, 最大=14.466, 最小=-14.139, 标准差=2.244
PO BID (2 times a day). lisinopril 5 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). atorvastatin 80 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). metolazone 5 mg Tablet Sig: One (1) Tablet PO DAILY...

**HARD_NEGATIVE (ID: 843949)**
Logit统计: 平均=-1.641, 最大=14.955, 最小=-14.879, 标准差=2.213
Sig: One (1) Tablet PO DAILY (Daily). aspirin 81 mg Tablet, Chewable Sig: One (1) Tablet, Chewable PO DAILY (Daily). Disp:*30 Tablet, Chewable(s)* Refills:*2* clonidine 3 mg/24 hr Patch Weekly Sig: On...

**HARD_NEGATIVE (ID: 1226856)**
Logit统计: 平均=-1.525, 最大=13.967, 最小=-13.747, 标准差=2.270
One (1) Tablet PO DAILY (Daily). - cholecalciferol (vitamin D3) 400 unit Tablet Sig: Two (2) Tablet PO once a day. -calcium carbonate 200 mg (500 mg) Tablet, Chewable Sig: One (1) Tablet, Chewable PO ...

**HARD_NEGATIVE (ID: 1579255)**
Logit统计: 平均=-1.545, 最大=14.593, 最小=-14.293, 标准差=2.237
100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day). Acetaminophen 500 mg Tablet Sig: Tablets PO every eight (8) hours as needed for pain, fever. Amlodipine 5 mg Tablet Sig: Two (2) Tablet PO D...

**HARD_NEGATIVE (ID: 1388255)**
Logit统计: 平均=-1.544, 最大=14.398, 最小=-14.147, 标准差=2.289
day). pravastatin 20 mg Tablet Sig: One (1) Tablet PO at bedtime. digoxin 125 mcg Tablet Sig: One (1) Tablet PO DAILY (Daily). calcium carbonate 200 mg calcium (500 mg) Tablet, Chewable Sig: One (1) T...

**HARD_NEGATIVE (ID: 452637)**
Logit统计: 平均=-1.560, 最大=14.600, 最小=-14.303, 标准差=2.247
mg Tablet Sig: 5 Tablets PO DAILY (Daily). Simvastatin 10 mg Tablet Sig: Two (2) Tablet PO DAILY (Daily). Warfarin 1 mg Tablet Sig: One (1) Tablet PO EVERY OTHER DAY (Every Other Day). Magnesium Oxide...

**HARD_NEGATIVE (ID: 565169)**
Logit统计: 平均=-1.551, 最大=14.146, 最小=-13.937, 标准差=2.217
Tablet Sig: One (1) Tablet PO 5X/WEEK (). Terazosin 5 mg Capsule Sig: One (1) Capsule PO HS (at bedtime). Fluticasone-Salmeterol 250-50 mcg/Dose Disk with Device Sig: One (1) Disk with Device Inhalati...

**HARD_NEGATIVE (ID: 866545)**
Logit统计: 平均=-1.467, 最大=13.480, 最小=-13.365, 标准差=2.089
One (1) Tablet, Chewable PO DAILY (Daily). Plavix 75 mg Tablet Sig: One (1) Tablet PO once a day. acyclovir 200 mg Capsule Sig: Two (2) Capsule PO Q12H (every 12 hours). amiodarone 200 mg Tablet Sig: ...

**HARD_NEGATIVE (ID: 659244)**
Logit统计: 平均=-1.537, 最大=14.499, 最小=-14.202, 标准差=2.240
hr Sig: One (1) Patch 24 hr Transdermal DAILY (Daily). senna 6 mg Tablet Sig: One (1) Tablet PO BID (2 times a day) as needed for constipation. docusate sodium 100 mg Capsule Sig: One (1) Capsule PO B...

**HARD_NEGATIVE (ID: 173958)**
Logit统计: 平均=-1.547, 最大=14.532, 最小=-14.451, 标准差=2.164
Delayed Release(E.C.) PO BID (2 times a day). Mycophenolate Mofetil 500 mg PO DAILY x 5 days. Then 500 mg BID x 7 days-starting on . Then 1000 mg qAM, 500 mg qPM x 7 days-starting . Then 1000 mg BID o...

**HARD_NEGATIVE (ID: 1532334)**
Logit统计: 平均=-1.476, 最大=14.143, 最小=-13.802, 标准差=2.227
DAILY (Daily). clopidogrel 75 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). atorvastatin 80 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Imdur 120 mg Tablet Extended Release 24 hr Sig: One (1) Ta...

**HARD_NEGATIVE (ID: 179516)**
Logit统计: 平均=-1.605, 最大=14.844, 最小=-14.798, 标准差=2.250
day). Metronidazole 500 mg Tablet Sig: One (1) Tablet PO Q8H (every 8 hours) for 10 days. Acetaminophen 325 mg Tablet Sig: One (1) Tablet PO Q6H (every 6 hours) as needed for pain. Aspirin 325 mg Tabl...

**HARD_NEGATIVE (ID: 912799)**
Logit统计: 平均=-1.594, 最大=14.963, 最小=-14.783, 标准差=2.356
needed). Metolazone 5 mg Tablet Sig: One (1) Tablet PO once a day. Metoprolol Succinate 25 mg Tablet Sustained Release 24 hr Sig: One (1) Tablet Sustained Release 24 hr PO DAILY (Daily). Morphine 15 m...

**HARD_NEGATIVE (ID: 172756)**
Logit统计: 平均=-1.477, 最大=13.858, 最小=-13.634, 标准差=2.150
Succinate 50 mg Tablet Sustained Release 24 hr Sig: One (1) Tablet Sustained Release 24 hr PO DAILY (Daily). Simvastatin 10 mg Tablet Sig: Two (2) Tablet PO DAILY (Daily). Ferrous Gluconate 325 mg (5 ...

**HARD_NEGATIVE (ID: 149189)**
Logit统计: 平均=-1.521, 最大=14.723, 最小=-14.478, 标准差=2.274
DAILY (Daily). Atorvastatin 80 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Lisinopril 5 mg Tablet Sig: 5 Tablet PO DAILY (Daily). Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 time...

**HARD_NEGATIVE (ID: 1380161)**
Logit统计: 平均=-1.565, 最大=14.436, 最小=-14.364, 标准差=2.190
DAILY (Daily). acetaminophen 500 mg Tablet Sig: Tablets PO Q6H (every 6 hours) as needed for pain. simvastatin 40 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). oxycodone 5 mg Tablet Sig: Tablets PO ...

**HARD_NEGATIVE (ID: 1432588)**
Logit统计: 平均=-1.536, 最大=14.598, 最小=-14.140, 标准差=2.270
Tablet PO TID (3 times a day) as needed for pain. Lisinopril 20 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). Metformin 500 mg Tablet Sig: One (1) Tablet PO BID (2 times a day). Fluticasone 50 mcg/A...

#### 相关问题:

1. Are medications listed with specific quantities and refill information? (来源文档: 16125)
2. Are there multiple medications listed for constipation management? (来源文档: 16125)
3. Are the medications prescribed primarily for symptom relief or chronic management? (来源文档: 16125)
4. Does the report mention that interventions led to symptom relief? (来源文档: 16125)
5. Are vitamins or supplements included in the list of medications? (来源文档: 16125)
6. Is there a specific schedule for administration included in the article? (来源文档: 16125)
7. Are there specific indications for when to take the medications? (来源文档: 16125)
8. Does the article contain instructions for inhalation or nebulization? (来源文档: 16125)
9. Is the article prescription-based? (来源文档: 16125)
10. Are medications listed with specific quantities and refill information? (来源文档: 46028)

## 2. 问题质量分析（真实模型logit分数）

### 随机文档 1 (ID: 1113852)

**文档内容**: Sternocleidomastoid and trapezius normal bilaterally. XII: Tongue midline without fasciculations. Motor: Normal bulk and tone bilaterally. No abnormal movements, tremors. Strength full power throughou...

#### 最高Logit分数问题:
- Does the article describe a medical condition related to neurological symptoms? (分数: 3.081)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 3.102)
- Is there a description of symptoms experienced by the patient? (分数: 3.102)
- Is there no mention of chest wall deformities or abnormalities? (分数: 3.406)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 3.433)
- Are there no reports of accessory muscle use during breathing? (分数: 3.479)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 3.551)
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.745)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.615)
- Is the article narrated in a professional medical tone? (分数: 14.324)

#### 中等Logit分数问题:
- Is there a mention of successful management of hyponatremia or hypernatremia? (分数: -1.596)
- Are vital signs recorded that indicate instability or critical condition upon arrival to the MICU? (分数: -1.593)
- Is there a report of an appropriate hemodynamic response to infusion? (分数: -1.591)
- Is there a plan described for a surgical weight loss procedure? (分数: -1.581)
- Does the article discuss effective strategies for managing dysphagia? (分数: -1.579)
- Is there evidence of scheduled screening tests or evaluations in the article? (分数: -1.577)
- Does the discharge instruction include avoiding heavy lifting for two weeks?   (分数: -1.576)
- Is there a discussion of specific medical conditions that can raise stroke risk? (分数: -1.574)
- Does the article mention "No Known Allergies" or "No Adverse Drug Reactions"? (分数: -1.573)
- Are there findings indicating the presence of significant arterial disease or occlusion? (分数: -1.571)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -14.328)
- Does the article refer to any educational resources provided to the patient regarding opioid therapy? (分数: -5.716)
- Does the article mention diabetic ketoacidosis (DKA)? (分数: -5.643)
- Does the article reference the need for repeat interventions or procedures related to biliary treatment? (分数: -5.611)
- Does the article contain a prescription for Albuterol? (分数: -5.499)
- Is there a history of an abdominal aortic aneurysm mentioned in the article? (分数: -5.409)
- Does the article mention a diagnosis of diffuse large B-cell lymphoma (DLBCL)? (分数: -5.382)
- Is there a documented complaint of flank pain in the article?   (分数: -5.292)
- Does the article include a scheduled TAVR procedure? (分数: -5.285)
- Does the article include a history of ERCP or related procedures? (分数: -5.250)

### 随机文档 2 (ID: 330160)

**文档内容**: Name: Unit No: Date of Birth: Sex: F Allergies: Metformin / Troglitazone / Statins-Hmg-Coa Reductase Inhibitors Attending: . Chief Complaint: GI bleed Major Surgical or Invasive Procedure: None Histor...

#### 最高Logit分数问题:
- Is there no evidence of neglect or apraxia documented in the article? (分数: 3.578)
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.677)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 4.068)
- Is the patient reported to have complex or chronic health issues? (分数: 4.145)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 4.197)
- Are there no reports of accessory muscle use during breathing? (分数: 4.322)
- Does the case history include a comprehensive assessment of the patient's issues? (分数: 4.380)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 4.537)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 5.145)
- Is the article narrated in a professional medical tone? (分数: 15.807)

#### 中等Logit分数问题:
- Does the article indicate recent hospitalization for complications related to alcohol use? (分数: -1.431)
- Is the stool consistency described as soft or formed without mention of diarrhea from an infectious agent? (分数: -1.431)
- Does the article mention "No Known Allergies" or "No Adverse Drug Reactions"? (分数: -1.426)
- Are antibiotics prescribed for a specific urological infection in the article? (分数: -1.425)
- Does the article mention elevated liver function tests (LFTs)? (分数: -1.423)
- Is there a clear history of memory loss or confusion? (分数: -1.422)
- Does the article mention a diagnosis of pulmonary embolism (PE)? (分数: -1.419)
- Does the patient have elevated troponin levels? (分数: -1.413)
- Has the patient undergone surgical intervention for lung-related issues? (分数: -1.408)
- Does the pathology indicate manageable liver conditions rather than advanced disease? (分数: -1.408)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.244)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.622)
- Is the use of ice packs or heating pads mentioned for symptomatic relief? (分数: -6.609)
- Does the article mention the use of deep brain stimulation as a treatment? (分数: -6.283)
- Was the patient initiated on IL-2 therapy during their hospitalization? (分数: -6.150)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.123)
- Is it mentioned that the patient can shower after a certain period? (分数: -5.858)
- Does the article mention that appetite suppression may improve over time? (分数: -5.838)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -5.810)
- Does the article mention keeping the incision dry until removal of sutures or staples? (分数: -5.775)

### 随机文档 3 (ID: 119285)

**文档内容**: - If patient's SOB, anemia significantly worsens, consider w/u for bone marrow disorder (e.g. MDS-like picture). - Recommend outpatient iron infusions - Repeat CBC in x1 week to ensure stability - dis...

#### 最高Logit分数问题:
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 3.307)
- Is there a focus on managing serious health symptoms in the article? (分数: 3.391)
- Is there no mention of chest wall deformities or abnormalities? (分数: 3.394)
- Is there no evidence of neglect or apraxia documented in the article? (分数: 3.400)
- Is there no mention of joint pains or myalgias? (分数: 3.418)
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.710)
- Are there no reports of accessory muscle use during breathing? (分数: 3.947)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 3.963)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.237)
- Is the article narrated in a professional medical tone? (分数: 14.924)

#### 中等Logit分数问题:
- Was assessment for rehabilitation or rehab services considered? (分数: -1.592)
- Did the patient demonstrate the ability to void and move bowels spontaneously? (分数: -1.591)
- Did the article mention that the patient's pain was controlled with oral medications? (分数: -1.589)
- Is there a discussion of discontinuation of psychiatric medications? (分数: -1.588)
- Is the patient's code status clearly outlined? (分数: -1.584)
- Is there mention of medications for respiratory conditions? (分数: -1.584)
- Did the discharge medications include an insulin regimen? (分数: -1.577)
- Is there a mention of alcohol use or its potential impact on the patient's condition? (分数: -1.565)
- Is the patient's general condition described as well-nourished or well-developed? (分数: -1.565)
- Does the article report a positive evaluation by a consulting specialist? (分数: -1.559)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.012)
- Does the article mention the use of deep brain stimulation as a treatment? (分数: -7.531)
- Is there a detailed description of the seizure semiology included in the article?   (分数: -7.216)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.527)
- Does the article describe a patient with pelvic organ prolapse? (分数: -6.329)
- Is there a mention of the patient receiving Electroconvulsive Therapy (ECT) or similar treatments? (分数: -6.183)
- Is the chief complaint related to syncope or near syncope? (分数: -6.028)
- Does the patient exhibit any postictal symptoms after seizures?   (分数: -5.988)
- Does the article mention surgical intervention for back pain? (分数: -5.964)
- Is there a plan described for a surgical weight loss procedure? (分数: -5.926)

### 随机文档 4 (ID: 1064993)

**文档内容**: stool incontinence. Wound care consult was called and Gaymar Overlay and Corticaid Ointment were ordered for the patient. The patient continued to refuse other treatment of her GI disturbances and wou...

#### 最高Logit分数问题:
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.514)
- Are there no reports of accessory muscle use during breathing? (分数: 3.522)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 3.601)
- Does the article describe a treatment plan involving medications for a health condition? (分数: 3.677)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 3.711)
- Is there no mention of chest wall deformities or abnormalities? (分数: 3.746)
- Is a healthcare provider involved in the prescribing of the medications? (分数: 3.904)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 4.275)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.658)
- Is the article narrated in a professional medical tone? (分数: 15.699)

#### 中等Logit分数问题:
- Does the article report the patient experiencing fatigue? (分数: -1.616)
- Does the article mention the continuation or adjustment of antifungal medication? (分数: -1.612)
- Is there a record of negative results for urinary tract infections? (分数: -1.609)
- Were conservative treatments explored before proceeding to surgery? (分数: -1.609)
- Does the article emphasize the importance of hydration?   (分数: -1.604)
- Is there mention of a negative work-up for identifiable vascular or neurological causes? (分数: -1.603)
- Is there a mention of ongoing monitoring or guidance for ostomy care? (分数: -1.596)
- Does the article detail blood work results indicating significant findings related to blood counts? (分数: -1.595)
- Does the article describe the patient expressing regret or a desire for help? (分数: -1.594)
- Did the article include a thorough review of systems that were mostly negative? (分数: -1.594)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.285)
- Is there a reference to procedures that utilize robotic assistance for surgery? (分数: -7.090)
- Does the patient's history include a nephrectomy for a renal mass? (分数: -6.620)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.607)
- Does the article mention a diagnosis of diffuse large B-cell lymphoma (DLBCL)? (分数: -6.551)
- Are there any signs of mitral regurgitation mentioned? (分数: -6.500)
- Is the use of ice packs or heating pads mentioned for symptomatic relief? (分数: -5.910)
- Is there a reference to multiple vessel disease in the cardiac assessment? (分数: -5.883)
- Is there a description of multiple lesions or nodules in the lung? (分数: -5.679)
- Was the patient discharged due to expiration? (分数: -5.628)

### 随机文档 5 (ID: 168005)

**文档内容**: Name: Unit No: Date of Birth: Sex: F /GYNECOLOGY Allergies: penicillin G / ampicillin Attending: Chief Complaint: Rectocele Major Surgical or Invasive Procedure: Posterior Colporrhaphy, Cystoscopy, Tr...

#### 最高Logit分数问题:
- Is there no mention of chest wall deformities or abnormalities? (分数: 3.269)
- Does the case history include a comprehensive assessment of the patient's issues? (分数: 3.486)
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.595)
- Are there no reports of accessory muscle use during breathing? (分数: 3.731)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 3.795)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 3.837)
- Is the article associated with gynecological issues or complaints? (分数: 4.072)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 4.240)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.610)
- Is the article narrated in a professional medical tone? (分数: 14.827)

#### 中等Logit分数问题:
- Is there a mention of treatment modalities such as chemotherapy or transfusions? (分数: -1.531)
- Are there reported symptoms of severe depression or anxiety? (分数: -1.527)
- Is there mention of using Tylenol as a first-line pain medication? (分数: -1.527)
- Are the blood glucose levels consistently below 200 mg/dL throughout the article? (分数: -1.526)
- Is there mention of positive trends in important lab values during hospitalization? (分数: -1.525)
- Does the discharge summary indicate a diagnosis requiring anticoagulation? (分数: -1.515)
- Does the article mention the presence of lower limb swelling? (分数: -1.508)
- Are there recommendations for lifestyle changes to enhance recovery? (分数: -1.507)
- Is there a statement about the patient's willingness to follow treatment plans? (分数: -1.491)
- Is there a description of urine being described as red or bloody? (分数: -1.489)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -14.642)
- Was the patient initiated on IL-2 therapy during their hospitalization? (分数: -6.978)
- Does the article involve a patient undergoing high-dose methotrexate treatment? (分数: -6.189)
- Was the patient discharged due to expiration? (分数: -6.097)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -6.084)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.063)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.042)
- Was LDL documented for the patient? (分数: -5.871)
- Does the article mention that appetite suppression may improve over time? (分数: -5.710)
- Is there a focus on a treatment plan involving TACE for liver-related conditions? (分数: -5.541)

### 随机文档 6 (ID: 389703)

**文档内容**: Name: Unit No: Date of Birth: Sex: F Allergies: Sulfa (Sulfonamide Antibiotics) / Codeine / Ibuprofen / Ancef / Bactrim / Betadine Viscous Gauze / Cipro I.V. / Flagyl Attending: . Chief Complaint: s/p...

#### 最高Logit分数问题:
- Is there a focus on managing serious health symptoms in the article? (分数: 3.408)
- Is the patient reported to have complex or chronic health issues? (分数: 3.559)
- Is there no mention of chest wall deformities or abnormalities? (分数: 3.585)
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.704)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 4.114)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 4.143)
- Are there no reports of accessory muscle use during breathing? (分数: 4.303)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 4.707)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.937)
- Is the article narrated in a professional medical tone? (分数: 15.970)

#### 中等Logit分数问题:
- Is the patient able to tolerate a regular diet? (分数: -1.446)
- Does the article contain mentions of previous evaluations that were normal or unchanged? (分数: -1.444)
- Is there a discussion of discontinuation of psychiatric medications? (分数: -1.443)
- Is the patient described as being in good spirits after their medical intervention? (分数: -1.442)
- Does the article mention surgical intervention for back pain? (分数: -1.438)
- Is there documentation of a patient's treatment for a urinary tract infection (UTI)? (分数: -1.433)
- Are there prophylactic measures provided for infection or complications? (分数: -1.431)
- Is the discharge disposition stated as "Home" in the article? (分数: -1.429)
- Are there imaging findings suggesting new or worsening disease? (分数: -1.428)
- Does the article indicate an absence of carcinoma in the examined tissues? (分数: -1.418)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.270)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -7.098)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.441)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -6.394)
- Does the article involve a patient undergoing high-dose methotrexate treatment? (分数: -6.149)
- Does the article mention that appetite suppression may improve over time? (分数: -5.996)
- Was the patient initiated on IL-2 therapy during their hospitalization? (分数: -5.887)
- Is it mentioned that the patient can shower after a certain period? (分数: -5.861)
- Does the article mention keeping the incision dry until removal of sutures or staples? (分数: -5.748)
- Is there a focus on a treatment plan involving TACE for liver-related conditions? (分数: -5.741)

### 随机文档 7 (ID: 143699)

**文档内容**: dog outside. At that point, on speaking to her dog, she noticed changes in her pronunciation, which was confirmed by her neighbor, prompting presentation to the OSH for further evaluation. There, she ...

#### 最高Logit分数问题:
- Is the patient in a clinical or emergency setting? (分数: 3.610)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 3.846)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 3.978)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 4.025)
- Is there a description of symptoms experienced by the patient? (分数: 4.157)
- Is the patient reported to have complex or chronic health issues? (分数: 4.259)
- Does the case history include a comprehensive assessment of the patient's issues? (分数: 4.412)
- Are there no reports of accessory muscle use during breathing? (分数: 4.804)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 5.162)
- Is the article narrated in a professional medical tone? (分数: 15.516)

#### 中等Logit分数问题:
- Does the article document some form of intoxication prior to the patient's evaluation? (分数: -1.573)
- Is there a mention of appropriate pain management after the procedure? (分数: -1.571)
- Is there an emphasis on the importance of medication adherence in the discharge instructions? (分数: -1.565)
- Is there any significant abdominal or pelvic abnormality noted in the imaging?   (分数: -1.560)
- Are positive outcomes or ongoing management plans for heart health discussed? (分数: -1.554)
- Are there discharge instructions included for medication management? (分数: -1.554)
- Does the article state that the patient is experiencing a scheduled admission for treatment? (分数: -1.548)
- Is there mention of supportive measures, such as pain management or nutritional support? (分数: -1.547)
- Is there a statement about the patient's willingness to follow treatment plans? (分数: -1.547)
- Are all lymph nodes reported as negative for metastasis? (分数: -1.546)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.122)
- Is vaginal activity prohibited for a specified duration? (分数: -6.997)
- Does the article describe a patient with pelvic organ prolapse? (分数: -6.992)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.925)
- Is it mentioned that the patient can shower after a certain period? (分数: -6.722)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -6.638)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.570)
- Are there instructions about showering after surgery? (分数: -6.568)
- Does the article instruct to secure the Foley catheter at all times? (分数: -6.371)
- Is there a reference to procedures that utilize robotic assistance for surgery? (分数: -6.292)

### 随机文档 8 (ID: 1247879)

**文档内容**: and unclear "schizo-something." Reports patient has been "miserable" since they moved back from approximately years ago, with more severe depression she believes over the last two months related to hi...

#### 最高Logit分数问题:
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 3.693)
- Does the article describe a medical condition related to neurological symptoms? (分数: 3.748)
- Are there no reports of accessory muscle use during breathing? (分数: 3.949)
- Does the article highlight significant life stressors impacting the patient's mental health? (分数: 4.024)
- Is the patient reported to have complex or chronic health issues? (分数: 4.054)
- Is there evidence of interpersonal relationships affecting the person's mental state in the article? (分数: 4.126)
- Is there a description of symptoms experienced by the patient? (分数: 4.323)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 4.972)
- Is the patient's report consistent with a diagnosis of a major mental illness? (分数: 5.661)
- Is the article narrated in a professional medical tone? (分数: 14.356)

#### 中等Logit分数问题:
- Is there evidence of significant cardiac symptoms documented? (分数: -1.446)
- Does the discharge summary indicate a diagnosis requiring anticoagulation? (分数: -1.440)
- Is the operative extremity noted to be neurovascularly intact? (分数: -1.438)
- Is there evidence of successful management or treatment of a wound or ulcer? (分数: -1.431)
- Is there documentation of the patient being asymptomatic after a procedure? (分数: -1.425)
- Is there a report of a potentially resistant bacterial infection being considered in the treatment plan? (分数: -1.421)
- Is there evidence of a recent fall or injury reported in the article? (分数: -1.410)
- Are any specific medications added or adjusted during the stay to address infections? (分数: -1.407)
- Did the patient receive proper education regarding their care? (分数: -1.406)
- Is the patient's demographic information clearly provided in the article? (分数: -1.403)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -13.922)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.520)
- Is it mentioned that the patient can shower after a certain period? (分数: -6.166)
- Are there any signs of mitral regurgitation mentioned? (分数: -5.939)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -5.888)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -5.872)
- Is the use of ice packs or heating pads mentioned for symptomatic relief? (分数: -5.750)
- Are there instructions about showering after surgery? (分数: -5.670)
- Is the patient instructed to elevate their legs post-surgery? (分数: -5.656)
- Is there a remark confirming that the thoracic aorta is intact? (分数: -5.649)

### 随机文档 9 (ID: 142524)

**文档内容**: 1 Powder(s) by mouth once a day Discharge Medications: acyclovir 400 mg Tablet Sig: One (1) Tablet PO Q12H (every 12 hours). lorazepam 5 mg Tablet Sig: Tablets PO Q4H (every 4 hours) as needed for anx...

#### 最高Logit分数问题:
- Are the prescribed medications aimed at maintaining or improving health? (分数: 5.207)
- Is a healthcare provider involved in the prescribing of the medications? (分数: 5.212)
- Is there evidence of an active treatment plan outlined for the patient's diagnosis? (分数: 5.262)
- Does the article contain information about doses and frequencies for medications? (分数: 5.490)
- Are the discharge medications clearly noted? (分数: 5.597)
- Is the article prescription-based? (分数: 5.640)
- Is there a structured format used throughout the article for medication details? (分数: 5.667)
- Is there a specific schedule for administration included in the article? (分数: 6.301)
- Are there specific indications for when to take the medications? (分数: 6.980)
- Is the article narrated in a professional medical tone? (分数: 13.604)

#### 中等Logit分数问题:
- Does the article mention a diagnosis related to a renal issue? (分数: -1.819)
- Is levothyroxine continued or adjusted for hypothyroidism in the article? (分数: -1.815)
- Did the patient receive a successful intervention for cardiac issues? (分数: -1.813)
- Is the patient being monitored for cancer-related symptoms? (分数: -1.812)
- Does the article refer to a complete hospital course related to a UTI? (分数: -1.811)
- Is the primary focus of the article on the management of alcohol withdrawal symptoms? (分数: -1.804)
- Was the patient's diet upgraded to regular or soft from a liquid diet? (分数: -1.799)
- Is there a clear indication of no pathogens identified in microbiology reports? (分数: -1.799)
- Are there notes indicating proper patient understanding and consent for procedures? (分数: -1.786)
- Was assessment for rehabilitation or rehab services considered? (分数: -1.786)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -13.661)
- Is there a reference to multiple vessel disease in the cardiac assessment? (分数: -7.362)
- Does the patient's history include a nephrectomy for a renal mass? (分数: -7.271)
- Is there a reference to procedures that utilize robotic assistance for surgery? (分数: -6.987)
- Are the patient's symptoms severe enough to warrant a code stroke activation? (分数: -6.630)
- Was the patient discharged due to expiration? (分数: -6.538)
- Is the chief complaint related to syncope or near syncope? (分数: -6.451)
- Are there any signs of mitral regurgitation mentioned? (分数: -6.407)
- Does the article mention that feelings and reactions are normal and should go away in a short time? (分数: -6.357)
- Is the primary complaint related to rest pain or claudication in the extremities? (分数: -6.271)

### 随机文档 10 (ID: 1416023)

**文档内容**: Name: Unit No: Date of Birth: Sex: F Allergies: No Known Allergies / Adverse Drug Reactions Attending: . Chief Complaint: hypertensive urgency Major Surgical or Invasive Procedure: none History of Pre...

#### 最高Logit分数问题:
- Is there no mention of active suicidal or homicidal ideation in the patient's evaluation? (分数: 3.581)
- Is there a focus on managing serious health symptoms in the article? (分数: 3.778)
- Does the case history include a comprehensive assessment of the patient's issues? (分数: 3.799)
- Did the medical team address the patient's primary health concerns appropriately? (分数: 3.897)
- Is there a clear presentation of a significant health concern or diagnosis? (分数: 4.195)
- Is the patient's primary complaint related to a specific medical issue rather than a vague or non-specific symptom? (分数: 4.199)
- Is the patient reported to have complex or chronic health issues? (分数: 4.245)
- Are there no reports of accessory muscle use during breathing? (分数: 4.396)
- Does the article describe a patient with a specific medical condition indicating a degree of seriousness? (分数: 5.258)
- Is the article narrated in a professional medical tone? (分数: 15.885)

#### 中等Logit分数问题:
- Are there indications that the patient is experiencing acute complications related to diabetes? (分数: -1.463)
- Is the operative extremity noted to be neurovascularly intact? (分数: -1.451)
- Is there evidence of active collaboration with the patient's family or support system? (分数: -1.450)
- Is the right ventricular chamber size normal? (分数: -1.450)
- Is there a clear and coherent family history with no significant hereditary issues reported? (分数: -1.447)
- Is there a diagnosis of jaundice present in the article? (分数: -1.447)
- Does the article note that the patient produced urine post-transplant? (分数: -1.445)
- Does the article mention a patient experiencing epistaxis (nosebleeds)? (分数: -1.431)
- Were conservative treatments explored before proceeding to surgery? (分数: -1.427)
- Does the article discuss a patient with symptoms related to inflammatory bowel disease? (分数: -1.426)

#### 最低Logit分数问题:
- Is there evidence of Legionella found in the article? (分数: -15.033)
- Does the article include recommendations for eating small, frequent meals? (分数: -6.719)
- Is it mentioned that the patient can shower after a certain period? (分数: -6.236)
- Does the article instruct to secure the Foley catheter at all times? (分数: -6.189)
- Is there a specified duration for activities like wearing stockings or using assistive devices? (分数: -6.136)
- Does the article refer to any educational resources provided to the patient regarding opioid therapy? (分数: -5.957)
- Is the use of ice packs or heating pads mentioned for symptomatic relief? (分数: -5.860)
- Does the article mention the use of deep brain stimulation as a treatment? (分数: -5.784)
- Is the patient instructed to elevate their legs post-surgery? (分数: -5.776)
- Is there guidance on gradually increasing activity levels after a procedure? (分数: -5.755)

