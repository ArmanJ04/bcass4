module.exports = {
  networks: {
    ganache: {
      host: "127.0.0.1",
      port: 7545,
      network_id: "*",
      gas: 8000000,
    },
  },
  compilers: {
    solc: {
      version: "0.8.21",
    }
  }
};
