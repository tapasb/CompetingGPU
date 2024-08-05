#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 3

__device__ int checkWin(int board[N][N], int player) {
     for (int i = 0; i < N; i++) {
          if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return 1;
          if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return 1;
     }
     if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return 1;
     if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return 1;
     return 0;
}

__global__ void randomMove(int board[N][N], int player, curandState *state) {
     int idx = threadIdx.x;
     if (idx == 0) {
          int x, y;
          do {
                x = curand(&state[idx]) % N;
                y = curand(&state[idx]) % N;
          } while (board[x][y] != 0);
          board[x][y] = player;
     }
}

__global__ void blockOpponentMove(int board[N][N], int player) {
     int opponent = (player == 1) ? 2 : 1;
     int idx = threadIdx.x;
     if (idx == 0) {
          for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                     if (board[i][j] == 0) {
                          board[i][j] = opponent;
                          if (checkWin(board, opponent)) {
                                board[i][j] = player;
                                return;
                          }
                          board[i][j] = 0;
                     }
                }
          }
          for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                     if (board[i][j] == 0) {
                          board[i][j] = player;
                          return;
                     }
                }
          }
     }
}

void printBoard(int board[N][N]) {
     for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
                printf("%d ", board[i][j]);
          }
          printf("\n");
     }
     printf("\n");
}

int main() {
     int board[N][N] = {0};
     int *d_board;
     cudaMalloc(&d_board, N * N * sizeof(int));
     curandState *d_state;
     cudaMalloc(&d_state, sizeof(curandState));
     curand_init(0, 0, 0, d_state);

     int currentPlayer = 1;
     int moves = 0;
     while (moves < N * N) {
          cudaMemcpy(d_board, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
          if (currentPlayer == 1) {
                randomMove<<<1, 1>>>(d_board, currentPlayer, d_state);
          } else {
                blockOpponentMove<<<1, 1>>>(d_board, currentPlayer);
          }
          cudaMemcpy(board, d_board, N * N * sizeof(int), cudaMemcpyDeviceToHost);
          printBoard(board);
          if (checkWin(board, currentPlayer)) {
                printf("Player %d wins!\n", currentPlayer);
                break;
          }
          currentPlayer = (currentPlayer == 1) ? 2 : 1;
          moves++;
     }
     if (moves == N * N) {
          printf("It's a draw!\n");
     }

     cudaFree(d_board);
     cudaFree(d_state);
     return 0;
}
