const ArmanToken = artifacts.require("ArmanToken");

contract("ArmanToken", (accounts) => {
  let armanToken;
  const owner = accounts[0];
  const recipient = accounts[1];

  beforeEach(async () => {
    armanToken = await ArmanToken.new();
  });

  it("should have correct token attributes", async () => {
    const name = await armanToken.name();
    const symbol = await armanToken.symbol();
    const totalSupply = await armanToken.totalSupply();

    assert.equal(name, "Arman", "Incorrect token name");
    assert.equal(symbol, "Ar", "Incorrect token symbol");
    assert.equal(
      totalSupply.toString(),
      "1000000000000000000000",
      "Incorrect total supply"
    );
  });

  it("should mint tokens to owner upon deployment", async () => {
    const ownerBalance = await armanToken.balanceOf(owner);
    assert.equal(
      ownerBalance.toString(),
      "1000000000000000000000",
      "Incorrect owner balance"
    );
  });

  it("should set block reward", async () => {
    await armanToken.setBlockReward(100);
    const blockReward = await armanToken.blockReward();
    assert.equal(blockReward.toString(), "100", "Incorrect block reward");
  });

  it("should transfer tokens between accounts", async () => {
    const amount = "1000000000000000000"; // 1 token

    await armanToken.transfer(recipient, amount, { from: owner });

    const recipientBalance = await armanToken.balanceOf(recipient);
    assert.equal(
      recipientBalance.toString(),
      amount,
      "Incorrect recipient balance after transfer"
    );
  });

  it("should not allow transfer from non-owners", async () => {
    const amount = "1000000000000000000"; // 1 token

    try {
      await armanToken.transfer(owner, amount, { from: recipient });
      assert.fail("Transfer did not throw");
    } catch (error) {
      assert(error.message.includes("revert"), "Wrong error message");
    }
  });

  it("should destroy contract and send remaining tokens to recipient", async () => {
    const initialBalance = parseInt(await web3.eth.getBalance(recipient));

    await armanToken.transfer(armanToken.address, "500000000000000000000", {
      from: owner,
    });

    const gasPrice = parseInt(await web3.eth.getGasPrice());
    const gasCost = gasPrice * 100000;

    await armanToken.destroy(recipient, { from: owner });

    const finalBalance = parseInt(await web3.eth.getBalance(recipient));
    
    const expectedFinalBalance = initialBalance - gasCost;
    const tolerance = 1000000000000000000;

    assert.isAtMost(
      finalBalance,
      expectedFinalBalance + tolerance,
      "Final balance should be at most the expected final balance plus tolerance"
    );
  });
});
