import { LayerFunction, ReverseFunction } from "../../autodiff/reverse-function";
import { NumTensor } from "../../tensor";


class LinearLayer extends LayerFunction {
    /** @type {NumTensor} */ #weights;

    /** 
     * @param {ReverseFunction} composite 
    */
    constructor(composite, inFeatures, outFeatures) {
        super(composite);

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        this.#weights = new NumTensor([outFeatures, inFeatures]);
    }

    forwards(vector) {
        return super.forwards(this.#weights.v_mul(vector));
    }
    
    backwards() {
        return this.#weights.permute(1, 0).v_mul(super.backwards());
    }
}
