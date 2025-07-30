curl -Ls https://astral.sh/uv/install.sh | sh

uv vnenv project

source project/bin/activate

uv init

uv add -r requirements.txt