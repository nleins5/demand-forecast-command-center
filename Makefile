.PHONY: install run clean

install:
	python -m pip install -e .

run:
	python -m demand_forecast

clean:
	rm -rf artifacts data/daily_demand.csv

