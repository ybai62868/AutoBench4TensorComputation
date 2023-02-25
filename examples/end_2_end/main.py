from engine import engine_launch



def main():
	executor = "--exec torch"
	model = "--model resnet18"
	engine_launch("{} {}".format(executor, model))


if __name__ == "__main__":
	main()
