import os
import re
import shutil

pattern_morphed_cropped_digital = re.compile(r'(morph-).*(_Cropped)(\.png)')
pattern_morphed_cropped_printedscanned = re.compile(r'(morph-).*(_PrintedAndScanned)(\.png)')
pattern_genuineformorphed_cropped_digital = re.compile(r'^(?!morph).*(Cropped)(\.png)')
pattern_genuineformorphed_cropped_printedscanned = re.compile(r'^(?!morph).*(PrintedAndScanned)(\.png)')
pattern_genuine_digital = re.compile(r'(TestImages).*\.png')

DB_MORPHEDDB_DIGITAL = "morpheddbdigital"
DB_MORPHEDDB_PRINTEDSCANNED = "morpheddbprintedscanned"
DB_BIOMETIX_FERET = "biometixferet"


def build_custom_db():
    for subdir, dirs, files in os.walk("../assets/db/originals/morphedDB"):
        for file in files:
            file_name = file
            full_file_name = os.path.join(subdir, file)
            dest = None
            if len(pattern_morphed_cropped_digital.findall(file_name)) > 0:
                print("MORPHED " + full_file_name)
                dest = "../assets/db/" + DB_MORPHEDDB_DIGITAL + "/original_uncropped/morphed/imgs/"
            # elif len(pattern_genuineformorphed_cropped_digital.findall(file_name)) > 0:
            #     print("GENUINE4MORPHED " + full_file_name)
            #     dest = "../assets/db/" + DB_MORPHEDDB_DIGITAL + "/original_uncropped/genuine4morphed/imgs/"
            # elif len(pattern_genuine_digital.findall(full_file_name)) > 0:
            #     print("GENUINE " + full_file_name)
            #     dest = "../assets/db/" + DB_MORPHEDDB_DIGITAL + "/original_uncropped/genuine/imgs/"

            # if len(pattern_genuineformorphed_cropped_printedscanned.findall(file_name)) > 0:
            #     dest = "../assets/db/" + DB_MORPHEDDB_PRINTEDSCANNED + "/original_uncropped/genuine4morphed/imgs"
            # elif len(pattern_morphed_cropped_printedscanned.findall(file_name)) > 0:
            #     dest = "../assets/db/" + DB_MORPHEDDB_PRINTEDSCANNED + "/original_uncropped/morphed/imgs"
            # elif len(pattern_genuine_digital.findall(full_file_name)) > 0:
            #     dest = "../assets/db/" + DB_MORPHEDDB_PRINTEDSCANNED + "/original_uncropped/genuine/imgs/"

            if dest is not None:
                if not os.path.exists(dest):
                    print(dest)
                    os.makedirs(dest)
                shutil.copy(full_file_name, dest)


def build_fvs():
    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyesnose/morphed-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyesnose/morphed")
    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine4morphed-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine4morphed")
    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine")

    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyeslip/morphed-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyeslip/morphed")
    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine4morphed-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine4morphed")
    os.system("luajit ./../openface/batch-represent/main.lua -outDir ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine-csv-rep -data ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine")


def build96eyesnose():
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/morphed/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/96eyesnose/morphed/imgs --size 96")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine4morphed/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine4morphed/imgs --size 96")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/96eyesnose/genuine/imgs --size 96")


def build256eyesnose():
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/morphed/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/256eyesnose/morphed/imgs --size 256")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine4morphed/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/256eyesnose/genuine4morphed/imgs --size 256")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine/imgs/ align outerEyesAndNose ../assets/db/" + CURRENT_DB + "/256eyesnose/genuine/imgs --size 256")


def build96eyeslip():
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/morphed/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/96eyeslip/morphed/imgs --size 96")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine4morphed/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine4morphed/imgs --size 96")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/96eyeslip/genuine/imgs --size 96")


def build256eyeslip():
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/morphed/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/256eyeslip/morphed/imgs --size 256")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine4morphed/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/256eyeslip/genuine4morphed/imgs --size 256")
    os.system("python ./../openface/util/align-dlib.py ../assets/db/" + CURRENT_DB + "/original_uncropped/genuine/imgs/ align innerEyesAndBottomLip ../assets/db/" + CURRENT_DB + "/256eyeslip/genuine/imgs --size 256")


def build_variations():
    build96eyesnose()
    build256eyesnose()
    build96eyeslip()
    build256eyeslip()
    pass


CURRENT_DB = DB_MORPHEDDB_DIGITAL

if __name__ == '__main__':
    # build_custom_db()
    # build_variations()
    # build_fvs()
    pass
