docker run \
	-it \
	--memory=64g \
	--memory-swap=64g \
	--cpuset-cpus=0-19 \
	--gpus '"device=0,1"' \
	--volume ~/lenta:/home/user/lenta \
	--workdir /home/user/lenta \
	-p 1488:6006 \
	lenta-hack
