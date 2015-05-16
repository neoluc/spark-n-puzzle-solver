from pyspark import SparkContext
import argparse
import Sliding

def solve_puzzle(width, height, output_path, slave_number):
	sc = SparkContext("local", "Slide")
	hash_solution = Sliding.board_to_hash(width, height, Sliding.solution(width, height))
	boards_rdd = sc.parallelize([(hash_solution, 0)])
	level = 0
	while True:
		level += 1
		frontier_rdd = boards_rdd.filter(lambda step: step[1] == level - 1)
		if frontier_rdd.isEmpty():
			break
		boards_rdd = frontier_rdd\
			.flatMap(lambda step: Sliding.children(width, height, Sliding.hash_to_board(width, height, step[0])))\
			.map(lambda board_hash: (Sliding.board_to_hash(width, height, board_hash), level))\
			.union(boards_rdd)\
			.reduceByKey(lambda step_level_a, step_level_b: min(step_level_a, step_level_b))\
			.partitionBy(slave_number)

	boards_rdd.map(lambda step: (step[1], Sliding.hash_to_board(width, height, step[0]))).sortByKey().coalesce(1).saveAsTextFile(output_path)
	sc.stop()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Return the solution graph.")
	parser.add_argument("-W", "--width", type = int, help = "width of puzzle")
	parser.add_argument("-H", "--height", type = int, help = "height of puzzle")
	parser.add_argument("-O", "--output_path", type = str, help = "output file path")
	parser.add_argument("-S", "--slave_number", type = int, help = "number of slaves in spark")
	args = parser.parse_args()
	solve_puzzle(args.width, args.height, args.output_path, args.slave_number)