from pyspark import SparkContext
import argparse
import Sliding

def solve_puzzle(width, height, output_path, slave_number):
	def hash_to_board(state):
		return Sliding.hash_to_board(width, height, state)
	def board_to_hash(board):
		return Sliding.board_to_hash(width, height, board)
	def get_children_boards(board):
		return Sliding.children(width, height, board)
	def get_solution_hash():
		return Sliding.board_to_hash(width, height, Sliding.solution(width, height))

	sc = SparkContext("local", "Slide")
	boards_rdd = sc.parallelize([(get_solution_hash(), 0)])
	current_level = 0
	while True:
		current_level += 1
		frontier_rdd = boards_rdd.filter(lambda (state, level): level == current_level - 1)
		frontier_rdd.persist()
		if frontier_rdd.isEmpty():
			break
		boards_rdd = frontier_rdd\
			.flatMap(lambda (state, level): get_children_boards(hash_to_board(state)))\
			.map(lambda state_board: (get_children_boards(state_board), current_level))\
			.union(boards_rdd)\
			.reduceByKey(lambda step_level_a, step_level_b: min(step_level_a, step_level_b))\
			.partitionBy(slave_number)

	boards_rdd\
		.map(lambda (state, level): (level, hash_to_board(state)))\
		.sortByKey()\
		.coalesce(1)\
		.saveAsTextFile(output_path)

	sc.stop()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Return the solution graph.")
	parser.add_argument("-W", "--width", type = int, help = "width of puzzle")
	parser.add_argument("-H", "--height", type = int, help = "height of puzzle")
	parser.add_argument("-O", "--output_path", type = str, help = "output file path")
	parser.add_argument("-S", "--slave_number", type = int, help = "number of slaves in spark")
	args = parser.parse_args()
	solve_puzzle(args.width, args.height, args.output_path, args.slave_number)