
import pytest
from core.portfolio.wallet import Wallet

def test_wallet_initial_balance():
    wallet = Wallet()
    print("🧪 TEST: Wallet initializes properly.")
    print("➡️ EXPECTED: Initial balance >= 0")
    print(f"✅ ACTUAL: Balance = {wallet.balance}")
    assert wallet.balance >= 0

def test_wallet_buy_increases_inventory_and_reduces_balance():
    wallet = Wallet(starting_balance=1000.0)
    wallet.buy(price=100.0, quantity=0.01)
    print("🧪 TEST: wallet.buy() behavior")
    print("➡️ EXPECTED: Inventory > 0 and Balance < 1000")
    print(f"✅ ACTUAL: Inventory = {wallet.inventory}, Balance = {wallet.balance}")
    assert wallet.inventory > 0
    assert wallet.balance < 1000.0

def test_wallet_sell_clears_inventory_and_updates_balance():
    wallet = Wallet(starting_balance=1000.0)
    wallet.buy(price=100.0, quantity=0.01)
    wallet.sell(price=110.0, quantity=0.01)
    print("🧪 TEST: wallet.sell() behavior")
    print("➡️ EXPECTED: Inventory = 0 and Balance > 1000 after profit")
    print(f"✅ ACTUAL: Inventory = {wallet.inventory}, Balance = {wallet.balance}")
    assert wallet.inventory == 0
    assert wallet.balance > 1000.0

def test_wallet_raises_on_insufficient_balance():
    wallet = Wallet(starting_balance=0.5)
    print("🧪 TEST: Buying with insufficient balance")
    print("➡️ EXPECTED: ValueError with 'Insufficient balance'")
    with pytest.raises(ValueError, match="❌ Insufficient balance") as excinfo:
        wallet.buy(price=100.0, quantity=0.01)
    print(f"✅ ACTUAL: {str(excinfo.value)}")

def test_wallet_raises_on_insufficient_inventory():
    wallet = Wallet(starting_balance=1000.0)
    print("🧪 TEST: Selling more than available inventory")
    print("➡️ EXPECTED: ValueError with 'Insufficient inventory'")
    with pytest.raises(ValueError, match="❌ Insufficient inventory") as excinfo:
        wallet.sell(price=110.0, quantity=0.01)
    print(f"✅ ACTUAL: {str(excinfo.value)}")

def test_wallet_get_state_dict_values():
    wallet = Wallet(starting_balance=1000.0)
    wallet.buy(price=100.0, quantity=0.01)
    state = wallet.get_state_dict(current_price=105.0)
    print("🧪 TEST: get_state_dict() returns key values")
    print("➡️ EXPECTED: Inventory > 0, Unrealized PnL > 0")
    print(f"✅ ACTUAL: Inventory = {state['inventory']}, Unrealized PnL = {state['unrealized_pnl']}")
    assert state["inventory"] > 0
    assert state["unrealized_pnl"] > 0

def test_wallet_get_performance_metrics():
    wallet = Wallet(starting_balance=1000.0)
    wallet.buy(price=100.0, quantity=0.01)
    wallet.sell(price=110.0, quantity=0.01)
    metrics = wallet.get_performance_metrics(current_price=110.0)
    print("🧪 TEST: get_performance_metrics() calculation")
    print("➡️ EXPECTED: Realized PnL > 0, Win Rate = 1.0, Trade Count = 2")
    print(f"✅ ACTUAL: PnL = {metrics['realized_pnl']}, Win Rate = {metrics['win_rate']}, Trade Count = {metrics['trade_count']}")
    assert metrics["realized_pnl"] > 0
    assert metrics["win_rate"] == 1.0
    assert metrics["trade_count"] == 2
