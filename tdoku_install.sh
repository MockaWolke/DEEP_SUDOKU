# setup.sh
git clone https://github.com/brotskydotcom/sudoku-generator.git
git clone https://github.com/t-dillon/tdoku.git

pip install sudoku-generator/
pip install -e .

cd tdoku/
unzip data.zip
bash BUILD.sh all