import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.switching import (
    GENERATOR_CHOICES,
    MAX_GENERATORS,
    SmartModelEngineSwitch,
    _selected_generator,
)


class GeneratorSelectionTests(unittest.TestCase):
    def test_plain_and_relabeled_dropdown_values_parse(self):
        self.assertEqual(_selected_generator("Generator 3"), 3)
        self.assertEqual(_selected_generator("3: Flux 1"), 3)
        self.assertEqual(_selected_generator("10: SDXL"), 10)
        self.assertEqual(_selected_generator("garbage"), 1)

    def test_values_saved_before_the_rename_still_select_the_same_slot(self):
        """A graph saved when the dropdown said "Engine 3" must still run slot 3.

        The stored widget value is plain text, so every workflow in the wild
        carries the old wording. Falling back to slot 1 would silently route a
        different model chain and produce a wrong picture with no error.
        """
        self.assertEqual(_selected_generator("Engine 3"), 3)
        self.assertEqual(_selected_generator("Engine 10"), 10)
        self.assertEqual(_selected_generator("Engine 1"), 1)

    def test_a_chain_name_containing_digits_uses_the_slot_number(self):
        # "2: Klein 9B" must be slot 2, not slot 9.
        self.assertEqual(_selected_generator("2: Klein 9B"), 2)
        self.assertEqual(_selected_generator("1: Z-Turbo"), 1)

    def test_number_override_wins_and_clamps(self):
        self.assertEqual(_selected_generator("Generator 3", 5), 5)
        self.assertEqual(_selected_generator("Generator 3", 99), MAX_GENERATORS)


class GeneratorSwitchNodeTests(unittest.TestCase):
    def test_only_the_selected_generator_is_requested_lazily(self):
        node = SmartModelEngineSwitch()
        self.assertEqual(
            node.check_lazy_status("2: SDXL", generator_1="unused", generator_2=None),
            ["generator_2"],
        )
        # Once the selected input is evaluated, nothing more is requested -
        # the other nine chains are never executed.
        self.assertEqual(
            node.check_lazy_status("2: SDXL", generator_1=None, generator_2="ready"),
            [],
        )

    def test_routes_the_selected_value_of_any_type(self):
        node = SmartModelEngineSwitch()
        payload = object()
        (result,) = node.route("Generator 2", generator_1="wrong", generator_2=payload)
        self.assertIs(result, payload)

    def test_unconnected_selection_reports_the_connected_generators(self):
        node = SmartModelEngineSwitch()
        with self.assertRaises(ValueError) as caught:
            node.route("Generator 4", generator_1="a", generator_3="c")
        self.assertIn("generator 4", str(caught.exception))
        self.assertIn("1, 3", str(caught.exception))

    def test_sockets_named_before_the_rename_still_route(self):
        """API-format prompts and un-migrated graphs send engine_N.

        The web extension renames the sockets when a graph loads, but a prompt
        submitted straight to /prompt never passes through it.
        """
        node = SmartModelEngineSwitch()
        payload = object()
        (result,) = node.route("Engine 2", engine_1="wrong", engine_2=payload)
        self.assertIs(result, payload)
        self.assertEqual(
            node.check_lazy_status("2: SDXL", engine_1="unused", engine_2=None),
            ["engine_2"],
        )

    def test_declares_ten_lazy_wildcard_inputs_and_wildcard_output(self):
        declared = SmartModelEngineSwitch.INPUT_TYPES()
        self.assertEqual(list(declared["required"]), ["generator"])
        generator_inputs = [
            name
            for name in declared["optional"]
            if name.startswith("generator_") and name != "generator_number"
        ]
        self.assertEqual(len(generator_inputs), MAX_GENERATORS)
        for name in generator_inputs:
            kind, options = declared["optional"][name]
            self.assertEqual(str(kind), "*")
            self.assertTrue(options.get("lazy"))
        self.assertEqual(str(SmartModelEngineSwitch.RETURN_TYPES[0]), "*")
        self.assertEqual(len(GENERATOR_CHOICES), MAX_GENERATORS)
        # Stock combo/type validation must be bypassed for relabeled values.
        self.assertTrue(SmartModelEngineSwitch.VALIDATE_INPUTS())

    def test_registered_in_the_node_menu_under_its_unchanged_type(self):
        """The class name is the saved-graph key and must never be renamed."""
        import nodes as node_pack

        self.assertIn("SmartModelEngineSwitch", node_pack.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            node_pack.NODE_DISPLAY_NAME_MAPPINGS["SmartModelEngineSwitch"],
            "Generator Switch (only selected runs)",
        )

    def test_nothing_user_facing_still_says_engine(self):
        declared = SmartModelEngineSwitch.INPUT_TYPES()
        visible = [str(GENERATOR_CHOICES)]
        visible.append(str(declared["required"]["generator"][1]))
        for name, (_kind, options) in declared["optional"].items():
            visible.append(name)
            visible.append(str(options.get("tooltip", "")))
        self.assertNotIn("engine", " ".join(visible).lower())


if __name__ == "__main__":
    unittest.main()
