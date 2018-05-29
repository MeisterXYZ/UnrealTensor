#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	if (argc != 5)
		return 1;

	// Read arguments from commandline
	const int N = std::stoi(argv[1]);
	const std::string label = argv[2];
	const std::string inPath = argv[3];
	const std::string outPath = argv[4];

	// Open in file
	std::ifstream inFile(inPath);
	if (!inFile.is_open())
		return 1;

	// Extract non-empty lines from in file
	std::vector<std::string> lines;
	std::string currentLine;
	while (std::getline(inFile, currentLine)) {
		if(currentLine.size() > 0)
			lines.push_back(currentLine);
	}

	// Build up batches of runs
	std::vector<std::string> runs;
	int pos = 0;
	while (static_cast<int>(lines.size()) - pos >= N) {
		std::string run;

		// Append x1, y1, ..., xN, yN
		for (int k = 0; k < N; ++k) {
			++pos;
			const auto line = lines[pos];
			if (line.size() > 0) {
				const auto lineSepPos = line.find_first_of(':');

				const auto x_str = line.substr(0, lineSepPos - 1);
				const auto y_str = line.substr(lineSepPos + 2);

				run += x_str + ", ";
				run += y_str + ", ";
			}
		}

		// Append label
		run += label;
		runs.push_back(run);
	}

	// Write in append mode to outfile
	std::ofstream outFile(outPath, std::ios::app);
	for (const auto& run : runs)
		outFile << run << std::endl;

	//// For testing purposes
	//std::cout << runs[0] << std::endl;
	//std::cout << runs[1] << std::endl;
	//std::cout << runs[2] << std::endl;

	return 0;
}
