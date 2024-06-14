#/bin/bash
parent_dir=$(dirname $(pwd))
pip3 uninstall tensorrt-llm
echo "==== ${parent_dir}"
python3 command_build_install.py
