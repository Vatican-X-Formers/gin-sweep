import glob
import os
import re
import shutil
import tarfile
import tempfile
import datetime


def get_current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S_%f")


def package_tfevents(models_dir, save_dir='', regex=None, dry_run=False):
    tfevent_paths = glob.glob(models_dir + '/**/*tfevents*', recursive=True)
    if regex:
        tfevent_paths = list(
            filter(lambda x: re.search(regex, x), tfevent_paths))

    def exp_name_from_event_path(event_path):
        path_parts = event_path.split('/')
        try:
            branch_name = path_parts[-4]
        except IndexError:
            branch_name = 'unknown'
        return f'{branch_name}@{path_parts[-3]}'

    with tempfile.TemporaryDirectory() as tmp_boards_dir:
        for path in tfevent_paths:
            exp_dir = f'{tmp_boards_dir}/{exp_name_from_event_path(path)}'
            if not dry_run:
                os.makedirs(exp_dir, exist_ok=True)
                shutil.copy(path, exp_dir)
            else:
                print(exp_dir, path)

        timestamp = get_current_timestamp()
        archive_name = f'{save_dir}/tensorboards_{timestamp}.tar.gz'

        if not dry_run:
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(tmp_boards_dir,
                        arcname='')

        return archive_name
