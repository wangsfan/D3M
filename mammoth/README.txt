An implementation of Continual Normalization (CN) on top of public implementation of the Dark Experience Replay method (https://github.com/aimagelab/mammoth).
To replicate the Split CIFAR 10 results with CN32 (32 groups), run:
./run.sh 0 cn32 seq-cifar10
To replicate the Split Tiny IMN results with CN32 (32 groups), run:
./run.sh 0 cn32 seq-tinyimg

There are three arguments in the ./run.sh file:
Argument 1: gpu id (usually set to 0 - the first gpu)
Argument 2: normalization layer, available options: cn4, cn8, cn32, cn64, bn (default)
Argument 3: benchmark, available options: seq-cifar10, seq-tinyimg
