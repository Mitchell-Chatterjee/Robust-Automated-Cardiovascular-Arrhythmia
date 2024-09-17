# The CODE-dataset: an annotated 12-lead ECG dataset contaning exams from more than a million patients

Dataset with annotated 12-lead ECG records. The exams were taken in 811 counties in the state of
Minas Gerais/Brazil by the Telehealth Network of Minas Gerais (TNMG) between 2010 and 2016. And organized by
the CODE (Clinical outcomes in digital electrocardiography) group.

-------

The dataset is described in the paper "Automatic diagnosis of the 12-lead ECG using a deep neural network". 
https://www.nature.com/articles/s41467-020-15432-4.

Citation:
```
Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
 Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
```

Bibtex:
```
@article{ribeiro_automatic_2020,
  title = {Automatic Diagnosis of the 12-Lead {{ECG}} Using a Deep Neural Network},
  author = {Ribeiro, Ant{\^o}nio H. and Ribeiro, Manoel Horta and Paix{\~a}o, Gabriela M. M. and Oliveira, 
   Derick M. and Gomes, Paulo R. and Canazart, J{\'e}ssica A. and Ferreira, Milton P. S. and Andersson, 
   Carl R. and Macfarlane, Peter W. and Meira Jr., Wagner and Sch{\"o}n, Thomas B. and Ribeiro, Antonio Luiz P.},
  year = {2020},
  volume = {11},
  pages = {1760},
  doi = {https://doi.org/10.1038/s41467-020-15432-4},
  journal = {Nature Communications},
  number = {1}
}
```

-----

## Content:

The folder contain:
- `annotations.csv`: The file `annotations.csv` is a column separated file containing basic patient attributes.
- The ECG waveforms in the wfdb format. In the file `RECORDS.txt`, there is a path to each of the records available.
  And, the compressed files `S(XXX)0000.tar.gz` contain exams with exam_id ranging 
  from (XXX)0000 to (XXX)9999. As an example the folder `S0120000.tar.gz` contain exams with exam id
  ranging from 120000 to 129999. More details are given below

### Annotations

The file `annotations.csv` is column separated file containing basic patient attributes: sex (M or F) and age; a field `id_exam` can be used to match ids from this file with the traces; a field `id_patient` which contain an unique identifier for the patient (a single patient can have multiple exams); the csv files also columns `1dAVb, RBBB, LBBB, SB, AF, ST` corresponding to weather the annomally have been detected in the ECG (`=1`) or not (`=0`).
The abnormalities are  abnormalities:

* 1st degree AV block (1dAVb);
* right bundle branch block (RBBB);
* left bundle branch block (LBBB);
* sinus bradycardia (SB);
* atrial fibrillation (AF); and,
* sinus tachycardia (ST).

We describe the full methodology to obtain this annotation of the abnormality in the [paper](https://www.nature.com/articles/s41467-020-15432-4) section "Methods/Training and validation set annotation". It also contain a field `date_exam` containing the date the exam was taken.

### Traces

The traces are available in the wfdb format and are contained in the folder `ecg-traces` and were generated using the python library: https://github.com/MIT-LCP/wfdb-python The ecg traces are divided into chunks with 10000 exams. Each of them compressed as a `.tar.gz`. All the entries have both a .hea and a .mat

The nomenclature used for each entry is `TNMG($X)_N($Y).hea/.mat `where `$X` correspond to the identifier of the exam, i.e., the field id_exam in the csv metadata. And `$Y` correspond to the record number. That is in the same exam multiple records from 7 to 10 seconds are taken in quick sucession.

The wfdb files contain more traces than the preprocessed file that is because, for in the same exam some times multiple tracings are recorded in sequence in a short interval of time, for the nature communication paper (and also in the above preprocessed `hdf5` file) we always consider only the first record, but here all traces are available and `$Y` can be used to identify which one.

As an example one can use the following python script to plot a sample from the data:

```
import ecg_plot  # Can be installed using pip
import wfdb
import matplotlib.pyplot as plt  # Can be installed using pip

#### Path of a WFDB entry ####
# See RECORDS.txt
PATH = 'S0000000/TNMG1_N1' 
##############################

record = wfdb.rdrecord(PATH)
ecg_plot.plot(record.p_signal.T, sample_rate=record.fs,
              lead_index=record.sig_name, style='bw', columns=2, 
              row_height=8)
# rm ticks
plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False)  # labels along the bottom edge are off
ecg_plot.show()
```


NOTE: The date of the header file contains the date the exam was taken. For most of the cases, this should coincide 
with the field "date_exam" in "annotations.csv". We have noticed, however, a few divergences. When this happens we
recommend the user to consider the field in "annotations.csv" as the correct one.

---

## Number of exams, patients and ECGs

The file `annotations.csv` contain 2,322,465 exams from 1,558,748 patients. For instance, you can use the following 
**python** command to check these numbers:

```
import pandas as pd

df = pd.read_csv('annotations.csv')
n_exams = len(df['id_exam'].unique())
n_patients = len(df['id_patient'].unique())

print(n_exams, n_patients)
```

The file ``RECORDS.txt`` contain 8200070 records taken during 2829840 exams. Hence, not all exams have corresponding 
annotation.  You can find each ones have annotation by matching the ids in the wfdb filename with the `annotation.csv` 
file.

```
import os

file_name = 'RECORDS.txt'
with open(file_name) as f:
    lines = f.readlines()
 # Check folders
ids = []
all_files = []
for p in lines:
    pp = p.strip()
    if pp:
        folder, file_name = os.path.split(pp)
        ids += [int(pp.split('TNMG')[-1].split('_N')[0])]
        all_files += [pp]
        
# Get folders and ids
ids = set(ids)  # remove repeated entries
print(len(ids), len(all_files))
```

---

## SaMi-Trop dataset

There are some patients in this dataset that intersect the SaMi-Trop dataset. The SaMi-Trop dataset is described 
in the publication bellow:

```
Cardoso CS, Sabino EC, Oliveira CDL, et alLongitudinal 
study of patients with chronic Chagas cardiomyopathy in Brazil 
(SaMi-Trop project): a cohort profile. BMJ Open 2016;6:e011181. 
doi: 10.1136/bmjopen-2016-011181
```

The id of all samitrop patients is contained in `samitrop_patients.txt`. See the snippet bellow to see how to remove the
samitrop patients from `annotations.csv`. This should result in a final dataset of 2,314,864 exams from 1,556,934 
patients. The generated file, `annotations_nosamitrop.csv`, contain the annotations after the samitrop patients
have been removed.

```
import pandas as pd
import numpy as np

df = pd.read_csv('annotations.csv')
samitrop_patients = np.loadtxt('samitrop_patients.txt', dtype=int)
df_nosamitrop = df[~np.isin(df['id_patient'], samitrop_patients)]
n_exams_nosamitrop = len(df_nosamitrop['id_exam'].unique())
n_patients_nosamitrop = len(df_nosamitrop['id_patient'].unique())
df_nosamitrop.to_csv('annotations_nosamitrop.csv', index=False)

print(n_exams_nosamitrop, n_patients_nosamitrop)
```

----

## Contact person:

To report issues or inconsistencies or request additional clinical information. Please contact:

- Antônio Horta Ribeiro 
- mail1: antonio.horta.ribeiro@it.uu.se
- mail2: antonior92@gmail.com
