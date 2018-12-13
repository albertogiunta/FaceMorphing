import os
import re
import shutil

TASK_TRAINING = "training"
TASK_TESTING = "testing"

IMG_DIGITAL = "digital"
IMG_PS = "ps"

DB_MORPHEDDB = "morphedDB"
DB_PMDB_DIG = "pmDB"
DB_PMDB_PS = "pmDB_ps"


def build_training_pmdb_dig():
    import random
    random.seed(42)
    pattern_pmdbdig_g4m = re.compile(r'^((?!TestImages).(?!morph))*(\.png)$')
    pattern_pmdbdig_mor = re.compile(r'(morph__).*(\.png)')
    pattern_pmdbdig_gen = re.compile(r'(TestImages).*(\.png)')

    max_morphs = 500
    num_morphs = 0

    for subdir, dirs, files in os.walk("../assets/db/originals/" + DB_PMDB_DIG):
        for file in files:
            file_name = file
            full_file_name = os.path.join(subdir, file)
            img_type = None

            if len(pattern_pmdbdig_mor.findall(file_name)) > 0 and num_morphs < max_morphs and random.randint(0, 9) == 0:
                num_morphs += 1
                print("MORPHED " + full_file_name)
                img_type = "mor"
            elif len(pattern_pmdbdig_gen.findall(full_file_name)) > 0:
                print("GENUINE " + full_file_name)
                img_type = "gen"
            elif len(pattern_pmdbdig_g4m.findall(file_name)) > 0:
                #     TODO it matches also genuines, needs to be fixed. Now it's ok since genuine matching comes first.
                print("GENUINE4MORPHED " + full_file_name)
                img_type = "gen"

            if img_type is not None:
                dest = "../assets/db/training/digital/" + DB_PMDB_DIG + "/raw/" + img_type + "/imgs/"
                move_file(dest, full_file_name)


def build_training_pmdb_ps():
    import random
    random.seed(42)
    pattern_pmdbdig_mor = re.compile(r'(morph__).*(\.png)')
    pattern_pmdbdig_gen = re.compile(r'^((?!morph)).*(\.png)$')

    max_morphs = 500
    num_morphs = 0

    for subdir, dirs, files in os.walk("../assets/db/originals/" + DB_PMDB_PS):
        for file in files:
            file_name = file
            full_file_name = os.path.join(subdir, file)
            img_type = None

            if len(pattern_pmdbdig_mor.findall(file_name)) > 0 and num_morphs < max_morphs and random.randint(0, 9) <= 3:
                num_morphs += 1
                print("MORPHED " + full_file_name)
                img_type = "mor"
            elif len(pattern_pmdbdig_gen.findall(file_name)) > 0:
                print("GENUINE " + full_file_name)
                img_type = "gen"

            if img_type is not None:
                dest = "../assets/db/training/ps/" + DB_PMDB_DIG + "/raw/" + img_type + "/imgs/"
                move_file(dest, full_file_name)


def build_testing_morpheddb_dig():
    pattern_morphed_cropped_digital = re.compile(r'(morph-).*(_Cropped)(\.png)')
    pattern_genuineformorphed_cropped_digital = re.compile(r'^(?!morph).*(Cropped)(\.png)')
    pattern_genuine_digital = re.compile(r'(TestImages).*\.png')

    for subdir, dirs, files in os.walk("../assets/db/originals/" + DB_MORPHEDDB):
        for file in files:
            file_name = file
            full_file_name = os.path.join(subdir, file)
            img_type = None

            if len(pattern_morphed_cropped_digital.findall(file_name)) > 0:
                print("MORPHED " + full_file_name)
                img_type = "mor"
            elif len(pattern_genuine_digital.findall(full_file_name)) > 0:
                print("GENUINE " + full_file_name)
                img_type = "gen"
            elif len(pattern_genuineformorphed_cropped_digital.findall(file_name)) > 0:
                print("GENUINE4MORPHED " + full_file_name)
                img_type = "gen"

            if img_type is not None:
                dest = "../assets/db/testing/digital/" + DB_MORPHEDDB + "/raw/" + img_type + "/imgs/"
                # move_file(dest, full_file_name)


def build_testing_morpheddb_ps():
    pattern_morphed_cropped_printedscanned = re.compile(r'(morph-).*(_PrintedAndScanned)(\.png)')
    pattern_genuineformorphed_cropped_printedscanned = re.compile(r'^(?!morph).*(PrintedAndScanned)(\.png)$')

    for subdir, dirs, files in os.walk("../assets/db/originals/" + DB_MORPHEDDB):
        for file in files:
            file_name = file
            full_file_name = os.path.join(subdir, file)
            img_type = None

            if len(pattern_genuineformorphed_cropped_printedscanned.findall(file_name)) > 0:
                print("GENUINE " + full_file_name)
                img_type = "gen"
            elif len(pattern_morphed_cropped_printedscanned.findall(file_name)) > 0:
                print("MORPHED " + full_file_name)
                img_type = "mor"

            if img_type is not None:
                dest = "../assets/db/testing/ps/" + DB_MORPHEDDB + "/raw/" + img_type + "/imgs/"
                move_file(dest, full_file_name)


def build_openface_variations():
    # images_type = ["mor", "gen", "g4m"]
    images_type = ["gen"]
    aligns = ["outerEyesAndNose", "innerEyesAndBottomLip"]
    dims = ["96", "256"]
    aligns_short = ["eyesnose", "eyeslip"]

    for dim in dims:
        for curr_align, align in enumerate(aligns):
            for type in images_type:
                command = "python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_TASK + "/" + CURRENT_IMG_FORMAT + "/" + CURRENT_DB + "/raw/" + type + "/imgs/ align " + align + " ../assets/db/" + CURRENT_TASK + "/" + CURRENT_IMG_FORMAT + "/" + CURRENT_DB + "/" + dim + aligns_short[curr_align] + "/" + type + "/imgs --size " + dim
                print(command)
                os.system(command)

                # if dim == "96":
                #     command = "luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_TASK + "/" + CURRENT_IMG_FORMAT + "/" + CURRENT_DB + "/" + dim + aligns_short[curr_align] + "/" + type +"-csv-rep -data ../assets/db/" + CURRENT_TASK + "/" + CURRENT_IMG_FORMAT + "/" + CURRENT_DB + "/" + dim + aligns_short[curr_align] + "/" + type
                #     print(command)
                #     os.system(command)


def move_file(dest, full_file_name):
    if not os.path.exists(dest):
        print(dest)
        os.makedirs(dest)
    shutil.copy(full_file_name, dest)


CURRENT_TASK = TASK_TESTING
CURRENT_IMG_FORMAT = IMG_DIGITAL
CURRENT_DB = "biometix"

if __name__ == '__main__':
    # build_training_pmdb_dig()
    # build_training_pmdb_ps()
    # build_testing_morpheddb_dig()
    # build_testing_morpheddb_ps()
    build_openface_variations()
    pass
