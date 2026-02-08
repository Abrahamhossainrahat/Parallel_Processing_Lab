#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

/* -------- MPI দিয়ে string পাঠানোর function -------- */
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 0, MPI_COMM_WORLD);
}

/* -------- MPI দিয়ে string receive করার function -------- */
string receive_string(int sender) {
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &status);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 0, MPI_COMM_WORLD, &status);
    string res(buf);
    delete[] buf;
    return res;
}

int main(int argc, char **argv) {

    // ===== MPI initialize =====
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // total process সংখ্যা

    if (argc < 2) {
        if (rank == 0)
            cout << "Usage: mpirun -np <p> ./wordcount input.txt\n";
        MPI_Finalize();
        return 0;
    }

    string filename = argv[1];
    vector<string> all_lines;

    // ===== Master process file পড়বে =====
    if (rank == 0) {
        ifstream in(filename);
        string line;
        while (getline(in, line)) {
            if (!line.empty())
                all_lines.push_back(line);
        }
        in.close();
    }

    // ===== Line ভাগ করা (chunk calculation) =====
    int total = all_lines.size();
    int chunk = (total + size - 1) / size;

    // ===== Master → Worker data পাঠানো =====
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            string part;
            for (int j = i * chunk; j < min(total, (i + 1) * chunk); j++)
                part += all_lines[j] + "\n";
            send_string(part, i);
        }
    }

    // ===== Local data receive =====
    vector<string> local_lines;
    if (rank == 0) {
        for (int i = 0; i < min(chunk, total); i++)
            local_lines.push_back(all_lines[i]);
    } else {
        string recv = receive_string(0);
        stringstream ss(recv);
        string line;
        while (getline(ss, line))
            local_lines.push_back(line);
    }

    // ===== Searching start time =====
    double start_time = MPI_Wtime();

    // ===== Local word count =====
    unordered_map<string, int> local_count;

    for (string &line : local_lines) {

        // line format: "MD. PARVEJ","018 12 477"
        // শুধু name অংশ নেওয়া
        int q1 = line.find('"');
        int q2 = line.find('"', q1 + 1);
        string name = line.substr(q1 + 1, q2 - q1 - 1);

        string word;
        stringstream ss(name);
        while (ss >> word) {

            // punctuation remove
            word.erase(remove_if(word.begin(), word.end(), ::ispunct), word.end());
            transform(word.begin(), word.end(), word.begin(), ::tolower);

            local_count[word]++;
        }
    }

    double end_time = MPI_Wtime();

    // ===== Worker → Master local result পাঠানো =====
    if (rank != 0) {
        string out;
        for (auto &p : local_count)
            out += p.first + " " + to_string(p.second) + "\n";
        send_string(out, 0);

        cout << "Process " << rank
             << " execution time: "
             << end_time - start_time << " seconds\n";
    }

    // ===== Master result collect + sort =====
    else {
        unordered_map<string, int> final_count = local_count;

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            stringstream ss(recv);
            string word;
            int cnt;
            while (ss >> word >> cnt)
                final_count[word] += cnt;
        }

        // map → vector (sorting-এর জন্য)
        vector<pair<string, int>> words(final_count.begin(), final_count.end());

        // ===== Frequency অনুযায়ী descending sort =====
        sort(words.begin(), words.end(),
             [](auto &a, auto &b) {
                 return a.second > b.second;
             });

        // ===== Output =====
        cout << "\nTop 10 frequent words:\n";
        for (int i = 0; i < min(10, (int)words.size()); i++)
            cout << words[i].first << " : " << words[i].second << endl;

        cout << "\nTotal execution time: "
             << end_time - start_time << " seconds\n";
    }

    // ===== MPI finish =====
    MPI_Finalize();
    return 0;
}
