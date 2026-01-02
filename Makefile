.PHONY: watch test test_mnist watch_mnist test_nn watch_nn test_models test_equivariance train_mnist train_baseline train_augmented train_e2_simple test_train evaluate evaluate_models evaluate_baseline evaluate_augmented evaluate_e2_simple

test:
	idris2 --exec main test/Test.idr

watch:
	watchexec -e idr -- idris2 -p test --exec main test/Test.idr

test_mnist:
	idris2 --exec main test/TestMNIST.idr

watch_mnist:
	watchexec -e idr -- idris2 -p test --exec main test/TestMNIST.idr

test_nn:
	idris2 --exec main test/TestNN.idr

watch_nn:
	watchexec -e idr -- idris2 -p test --exec main test/TestNN.idr

test_models:
	poetry run pytest test/models/test_models.py -v

test_equivariance:
	poetry run python test/models/test_equivariance.py

test_train:
	poetry run pytest test/rotational_mnist/test_train.py -v -s

train_mnist:
	poetry run python scripts/rotational_mnist/train.py

train_baseline:
	poetry run python scripts/rotational_mnist/train.py baseline

train_augmented:
	poetry run python scripts/rotational_mnist/train.py augmented

train_e2_simple:
	poetry run python scripts/rotational_mnist/train.py e2_simple

evaluate:
	poetry run python scripts/rotational_mnist/evaluate.py

evaluate_baseline:
	poetry run python scripts/rotational_mnist/evaluate.py baseline

evaluate_augmented:
	poetry run python scripts/rotational_mnist/evaluate.py augmented

evaluate_e2_simple:
	poetry run python scripts/rotational_mnist/evaluate.py e2_simple

