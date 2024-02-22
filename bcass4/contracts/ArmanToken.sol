// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract ArmanToken is ERC20 {
    address public owner;
    uint256 public blockReward; // Changed to public

    constructor() ERC20("Arman", "Ar") {
        owner = msg.sender;
        _mint(msg.sender, 1000 * 10 ** uint(decimals()));
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    function _mintMinerReward() internal {
        _mint(block.coinbase, blockReward);
    }

    function setBlockReward(uint256 amount) external onlyOwner {
        blockReward = amount;
    }

function destroy(address payable recipient) external onlyOwner {
    require(recipient != address(0), "Invalid recipient address");
    recipient.transfer(address(this).balance);
}
}
