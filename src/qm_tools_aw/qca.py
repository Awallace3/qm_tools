from qcportal import PortalClient
from pprint import pprint as pp
from qcelemental.models import Molecule
from qcportal.singlepoint import (
    SinglepointDataset,
    SinglepointDatasetEntry,
    QCSpecification,
)
from qcportal.manybody import (
    ManybodyDataset,
    ManybodyDatasetEntry,
    ManybodyDatasetSpecification,
    ManybodySpecification,
)
from typing import Union


def establish_client(url="http://localhost:7777", verify=False):
    """Establish a connection to the QC Portal client."""
    return PortalClient(url, verify=verify)


def init_singlepoint_dataset(client, ds_name: str, geoms: [[str, Molecule]]=None):
    """
    Initialize a dataset in the QC Portal client.
    If the dataset does not exist, it will be created.
    If it exists, it will be retrieved.
    :param client: PortalClient instance
    :param ds_name: Name of the dataset to be initialized
    :param geoms: List of Molecule objects to be added to the dataset
        Note: include extras when creating the Molecule objects
    """
    client_datasets = [i["dataset_name"] for i in client.list_datasets()]
    if ds_name not in client_datasets:
        ds = client.add_dataset("singlepoint", ds_name, f"Dataset to contain {ds_name}")
        print(f"Added {ds_name} as dataset")
    else:
        ds = client.get_dataset("singlepoint", ds_name)
        print(f"Found {ds_name} dataset, using this instead")
    if geoms:
        cnt = 0
        entry_list = []
        for n, (name, mol) in enumerate(geoms):
            print(n, name, mol)
            ent = SinglepointDatasetEntry(name=name, molecule=mol)
            entry_list.append(ent)
            # if len(entry_list) > 250:
            #     cnt += len(entry_list)
            #     ds.add_entries(entry_list)
            #     print(f"Added {len(entry_list)} molecules to dataset")
            #     entry_list = []
        if len(entry_list) > 0:
            cnt += len(entry_list)
            ds.add_entries(entry_list)
        print(f"Added {cnt} molecules to dataset")
    return ds


def create_singlepoint_dataset_specification(
    ds: SinglepointDataset,
    program: str = "psi4",
    driver: str = "energy",
    method: str = "sapt(dft)",
    basis: str = "aug-cc-pvtz",
    keywords: dict = {"scf_type": "df"},
    protocols: dict = {'stdout': True},
    specification_name: str = None,
    compute_tag: str = "hive",
):
    spec = QCSpecification(
        program=program,
        driver=driver,
        method=method,
        basis=basis,
        keywords=keywords,
        protocols=protocols,
    )
    if specification_name is None:
        ds.add_specification(name=f"psi4/{method}/{basis}", specification=spec)
    else:
        ds.add_specification(name=specification_name, specification=spec)
    ds.submit(tag=compute_tag)
    print(ds.status())
    return


def check_record_status(client, singlepoint_outputs, delete=False):
    for n, job in enumerate(singlepoint_outputs):
        if n % 2 == 0:
            continue
        pp(job)
        v = client.query_records(record_id=job)
        for i in v:
            print(i)
        for j in job:
            reason = client.get_waiting_reason(j)
            pp(reason)
        if delete:
            client.delete_records(job, soft_delete=False)
    return


def collect_ids(singlepoint_outputs):
    ids = []
    for n, job in enumerate(singlepoint_outputs):
        if n % 2 == 0:
            continue
        for j in job:
            ids.append(j)
    return ids


def create_singlepoints(
    client: PortalClient,
    geoms,
    method="sapt(dft)",
    basis="aug-cc-pvtz",
    keywords={
        "freeze_core": "True",
        "d_convergence": 8,
        "scf_type": "df",
        "mp2_type": "df",
        "SAPT_DFT_GRAC_COMPUTE": "SINGLE",
        "SAPT_DFT_FUNCTIONAL": "pbe0",
    },
):
    jobs = client.add_singlepoints(
        geoms,
        "psi4",
        driver="energy",
        method=method,
        basis=basis,
        keywords=keywords,
        tag="nvme",
    )
    return jobs
