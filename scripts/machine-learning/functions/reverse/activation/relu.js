import { ReverseFunction } from "../reverse.js";
import { NumTensor } from "../../../tensor.js";

/**
 * Rectified Linear Unit (ReLU) activation function.
 * 
 * Definition: `f(x) = max(0, x)`
 */
export class ReLU extends ReverseFunction {

    forwards(tensor) {
        const result = new NumTensor(tensor.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.max(0, tensor._data[i]);
        }

        return super.forwards(result);
    }

    backwards() {
        const result = new NumTensor(this.forwardsCache.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this.forwardsCache._data[i] > 1 ? 1 : 0;
        }

        return result.hadamard(super.backwards());
    }
}
