const ArmanToken = artifacts.require("ArmanToken");

module.exports = function(deployer) {
  deployer.deploy(ArmanToken);
};
