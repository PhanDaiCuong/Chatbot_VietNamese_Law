for file in script/*.sh; do
    bash "$file" &
done
wait
