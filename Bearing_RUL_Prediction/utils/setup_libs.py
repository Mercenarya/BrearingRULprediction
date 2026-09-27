import os,sys
import subprocess
import time

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(CURRENT,"..")
sys.path.append(ROOT)


def auto_set_up_libs(branch:str,libs_file:str):
    try:
        requirements = os.path.join(branch,libs_file)
        
        command_sheets_pip_upgrade = ["python.exe","-m","install","--upgrade","pip"]
        command_sheets_pip_install = ["pip","install","-r",libs_file]
        command_sheets_cache_remove = ["pip","cache","purge"]
        
        command_sheets = [
            command_sheets_pip_upgrade,
            command_sheets_pip_install,
            command_sheets_cache_remove
        ]
         
        if os.path.exists(requirements):
            print(f"[+]: Branch {requirements} existed.")
            for sheets in range(len(command_sheets)):
                result = subprocess.run(
                    command_sheets[sheets],
                    check=False,
                    capture_output=False,
                    text=False,
                    cwd=branch
                )
                time.sleep(1)
                if result.returncode == 0:
                    print("[+]: Phase set-up completed.")
                else:
                    print("[x]: Phase set-up failed")
        print("[o]: Task Completed.")
    except Exception as error:
        print("[x]: An error occured during set up libs")
        print("[!]:",error)
    finally:
        pass


def check_version(data:str):
    try:
        result = subprocess.run(
            ["pip","show",data],
            check=False,
            capture_output=False,
            text=False
        )
        if result.returncode == 0:
            print(f"[lib]: {data} is responsed.")
        else:
            print(f"[lib]: {data} is unavailable.")
    except Exception as error:
        print("[x]: An error occured during checking info of libs")
        print("[!]:",error)
    finally:
        pass

def check_libs(branch:str,libs_file:str):
    try:
        symbols = [">","="," ",".","\n"]
        requirements = os.path.join(branch,libs_file)
        with open(requirements,"r") as file:
            
            for obj in file:
                prc_data = [
                    x for x in obj if x not in [i for i in symbols] and not x.isdigit()
                ]
                result = "".join(prc_data)
                check_version(result)
                time.sleep(1)
    except Exception as error:
        print("[x]: An error occured during checking list of libs")
        print("[!]:",error)
    finally:
        pass
    
if __name__ == "__main__":
    # auto_set_up_libs(ROOT,"requirements.txt")
    check_libs(ROOT,"requirements.txt")