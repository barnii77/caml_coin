import unittest

from src.miner import blockchain

USE_CUDA = True
if USE_CUDA:
    blockchain.load_cuda_miner(
        "cuda_miner_templates/build/nonce128_intrinsic_uint128_random.ptx",
    )
    # print("launching with launch config", blockchain._cuda_miner_fixed_launch_config_kernel.launch_config)


class BlockchainTests(unittest.TestCase):
    def test_blockchain_raw1(self):
        import threading

        names_to_keys = {
            b"name1": blockchain.gen_key_pair(),
            b"name2": blockchain.gen_key_pair(),
            b"name4": blockchain.gen_key_pair(),
            b"name3": blockchain.gen_key_pair(),
            b"name5": blockchain.gen_key_pair(),
        }

        def name_to_private_public_key(name: bytes):
            return names_to_keys[name]

        init_balance = {
            name_to_private_public_key(name)[1]: value
            for name, value in {
                b"name1": 101,
                b"name2": 2,
                b"name4": 624,
                b"name3": 84,
                b"name5": 1000,
            }.items()
        }

        (
            in_tq,
            in_cq,
            in_bq,
            out_cq,
            out_bq,
            out_sq,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        terminate_event = threading.Event()
        use_cuda_miner_event = threading.Event()
        blockchain_args = (
            name_to_private_public_key(b"name5")[1],
            None,
            in_tq,
            in_cq,
            in_bq,
            out_cq,
            out_bq,
            out_sq,
            init_balance,
            terminate_event,
            None,
            use_cuda_miner_event,
        )

        blockchain_thread = threading.Thread(
            target=blockchain._mine,
            args=blockchain_args,
            kwargs={
                "valid_block_max_hash": 0x00000003FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
                "min_income_to_mine": 0,
            },
        )
        if USE_CUDA:
            use_cuda_miner_event.set()
        blockchain_thread.start()

        def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
            import time
            import random

            timestamp_bytes = (time.time_ns() + random.randrange(0, 200)).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[4], "little"
            )
            # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
            private_key_sender, public_key_sender = name_to_private_public_key(sender)
            _, public_key_receiver = name_to_private_public_key(receiver)
            amount_bytes = amount.to_bytes(
                blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN
            )
            transaction_fee_bytes = (0).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN
            )
            # private_key_obj = blockchain.deserialize_private_key(private_key_sender)
            signature_bytes = blockchain.sign_message(
                public_key_sender
                + public_key_receiver
                + amount_bytes
                + transaction_fee_bytes
                + timestamp_bytes,
                private_key_sender,
            )
            return blockchain.Transaction(
                public_key_sender,
                public_key_receiver,
                amount,
                0,
                timestamp_bytes,
                time.time_ns().to_bytes(8, "little"),
                signature_bytes,
            )

        def test_transaction(sender: bytes, receiver: bytes):
            transaction = get_test_transaction(sender, receiver, 10)
            transaction_bytes = transaction.to_bytes()
            in_tq.append(transaction_bytes)

        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name2", b"name1")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name2", b"name3")

        import time

        time.sleep(120)
        terminate_event.set()
        blockchain_thread.join()
        balances = {
            hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value
            for key, value in out_bq.pop().items()
        }
        balances_values = list(balances.values())
        self.assertEqual(balances_values, [11, 82, 624, 94, 1000])

    def test_blockchain_class1(self):
        import random

        names_to_keys = {
            b"name1": blockchain.gen_key_pair(),
            b"name2": blockchain.gen_key_pair(),
            b"name4": blockchain.gen_key_pair(),
            b"name3": blockchain.gen_key_pair(),
            b"name5": blockchain.gen_key_pair(),
        }

        def name_to_private_public_key(name: bytes):
            if name not in names_to_keys:
                names_to_keys[name] = blockchain.gen_key_pair()
            return names_to_keys[name]

        init_balance = {
            name_to_private_public_key(name)[1]: value
            for name, value in {
                b"name1": 101,
                b"name2": 2,
                b"name4": 624,
                b"name3": 84,
                b"name5": 1000,
            }.items()
        }

        chain_handler = blockchain.Miner(
            name_to_private_public_key(b"name5")[1],
            None,
            init_balance,
            blockchain.MiningConfig(
                valid_block_max_hash=0x00000003FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
                min_income_to_mine=0,
            ),
            "threading",
        )
        if USE_CUDA:
            chain_handler.use_cuda_miner()

        def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
            import time
            import random

            timestamp_bytes = (time.time_ns() + random.randrange(0, 200)).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[4], "little"
            )
            # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
            private_key_sender, public_key_sender = name_to_private_public_key(sender)
            _, public_key_receiver = name_to_private_public_key(receiver)
            amount_bytes = amount.to_bytes(
                blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN
            )
            transaction_fee_bytes = (0).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN
            )
            # private_key_obj = blockchain.deserialize_private_key(private_key_sender)
            signature_bytes = blockchain.sign_message(
                public_key_sender
                + public_key_receiver
                + amount_bytes
                + transaction_fee_bytes
                + timestamp_bytes,
                private_key_sender,
            )
            return blockchain.Transaction(
                public_key_sender,
                public_key_receiver,
                amount,
                0,
                timestamp_bytes,
                timestamp_bytes,
                signature_bytes,
            )

        def test_transaction(sender: bytes, receiver: bytes):
            transaction = get_test_transaction(sender, receiver, 10)
            transaction_bytes = transaction.to_bytes()
            chain_handler.add_transaction(transaction_bytes)

        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name2", b"name1")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name2", b"name3")

        import time

        time.sleep(120)
        _, balances = chain_handler.finish()
        balances = blockchain.deserialize_balances(balances)
        balances = {
            hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value
            for key, value in balances.items()
        }
        balances_values = list(balances.values())
        self.assertEqual(balances_values, [11, 82, 624, 94, 1000])

    def test_blockchain_class2(self):
        import random

        names_to_keys = {
            b"name1": blockchain.gen_key_pair(),
            b"name2": blockchain.gen_key_pair(),
            b"name4": blockchain.gen_key_pair(),
            b"name3": blockchain.gen_key_pair(),
            b"name5": blockchain.gen_key_pair(),
        }

        def name_to_private_public_key(name: bytes):
            if name not in names_to_keys:
                names_to_keys[name] = blockchain.gen_key_pair()
            return names_to_keys[name]

        init_balance = {
            name_to_private_public_key(name)[1]: value
            for name, value in {
                b"name1": 101,
                b"name2": 2,
                b"name4": 624,
                b"name3": 84,
                b"name5": 1000,
            }.items()
        }

        mining_config = blockchain.MiningConfig(
            valid_block_max_hash=0x00000003FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
            min_income_to_mine=0,
        )
        chain_handler = blockchain.Miner(
            name_to_private_public_key(b"name5")[1],
            None,
            init_balance,
            mining_config,
            "mp",
        )
        if USE_CUDA:
            chain_handler.use_cuda_miner()

        def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
            import time
            import random

            timestamp_bytes = (time.time_ns() + random.randrange(0, 200)).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[4], "little"
            )
            # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
            private_key_sender, public_key_sender = name_to_private_public_key(sender)
            _, public_key_receiver = name_to_private_public_key(receiver)
            amount_bytes = amount.to_bytes(
                blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN
            )
            transaction_fee_bytes = (0).to_bytes(
                blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN
            )
            # private_key_obj = blockchain.deserialize_private_key(private_key_sender)
            signature_bytes = blockchain.sign_message(
                public_key_sender
                + public_key_receiver
                + amount_bytes
                + transaction_fee_bytes
                + timestamp_bytes,
                private_key_sender,
            )
            return blockchain.Transaction(
                public_key_sender,
                public_key_receiver,
                amount,
                0,
                timestamp_bytes,
                time.time_ns().to_bytes(8, "little"),
                signature_bytes,
            )

        def test_transaction(sender: bytes, receiver: bytes, amount=10):
            transaction = get_test_transaction(sender, receiver, amount)
            transaction_bytes = transaction.to_bytes()
            chain_handler.add_transaction(transaction_bytes)

        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name1", b"name2")
        test_transaction(b"name2", b"name3")
        test_transaction(b"name2", b"name4")
        test_transaction(b"name1", b"name2")  # should fail (not enough coins)
        test_transaction(b"name1", b"name2")  # should fail (not enough coins)
        test_transaction(b"name1", b"name2")  # should fail (not enough coins)
        test_transaction(
            b"not real name", b"name3"
        )  # should fail (not real name has balance 0)
        test_transaction(
            b"name2", b"name2"
        )  # should fail (sender and receiver are the same)
        test_transaction(b"name2", b"name3", 100000)  # should fail (not enough coins)
        test_transaction(b"name2", b"name3", 0)  # should fail (amount is 0)

        # test with invalid signature
        transaction = get_test_transaction(b"name1", b"name2", 10)
        transaction.timestamp = random.randbytes(
            blockchain.Transaction.BYTE_COUNTS[4]
        )  # change uuid, now signature is invalid
        transaction_bytes = transaction.to_bytes()
        chain_handler.add_transaction(transaction_bytes)

        # test with duplicated uuid (send same transaction twice)
        transaction = get_test_transaction(b"name1", b"name2", 10)
        transaction_bytes = transaction.to_bytes()
        chain_handler.add_transaction(transaction_bytes)
        chain_handler.add_transaction(transaction_bytes)

        import time

        time.sleep(150)
        chain_bytes, balances = chain_handler.finish()
        balances = blockchain.deserialize_balances(balances)
        balances = {
            hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value
            for key, value in balances.items()
        }
        chain = blockchain.chain_from_bytes(chain_bytes)
        self.assertNotEqual(chain, None)
        self.assertTrue(
            blockchain.full_verify(
                chain,
                init_balance,
                mining_config.val_reward,
                mining_config.const_transaction_fee,
                mining_config.relative_transaction_fee,
                mining_config.valid_block_max_hash,
                mining_config.max_timestamp_now_diff,
            )[0]
        )
        self.assertEqual(blockchain.chain_len(chain), 3)
        balances_values = list(balances.values())
        self.assertEqual(balances_values, [1, 82, 634, 94, 1000])


if __name__ == "__main__":
    unittest.main()
